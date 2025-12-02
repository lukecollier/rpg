use std::{borrow::Cow, collections::VecDeque, time::Duration};

use avian3d::prelude::*;
use bevy::{
    asset::{Asset, Handle, RenderAssetUsages},
    camera::primitives::Aabb,
    color::palettes::css::*,
    gltf::GltfMeshName,
    image::{Image, ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    math::bounding::{Aabb2d, BoundingVolume},
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial, wireframe::Wireframe,
    },
    platform::collections::{HashMap, HashSet},
    prelude::*,
    reflect::TypePath,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
            StorageTextureAccess, TextureFormat, TextureUsages, UniformBuffer,
            binding_types::{texture_storage_2d, uniform_buffer},
        },
        renderer::{RenderDevice, RenderQueue},
        storage::ShaderStorageBuffer,
        texture::GpuImage,
    },
    scene::SceneInstanceReady,
    shader::{PipelineCacheError, ShaderRef},
};

// todo: Move these into terrain settings, we can have TerrainPlugin init the resource
pub const CHUNK_HALF_SIZE: f32 = 128.;
pub const CHUNK_SIZE: f32 = CHUNK_HALF_SIZE * 2.;
pub const TERRAIN_SIZE: f32 = 2048. * 2.;
// todo: This need's to be the number of vertices in the x y of all our high detail chunks I GUESS
const HEIGHT_MAP_SIZE: u32 = 256;
const TERRAIN_MAX_HEIGHT: f32 = 1024.;

// After how many seconds of changes do we trigger a gpu readback for heightmaps
const HEIGHTMAP_DEBOUNCE_SECS: u64 = 1;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainRenderLabel;

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_plugins((
            MaterialPlugin::<BlendMaterial>::default(),
            MaterialPlugin::<TerrainMaterial>::default(),
        ))
        .insert_resource(TerrainSettings::default())
        .add_systems(
            Startup,
            (
                startup_textures,
                startup_test_tree,
                TerrainGpuImages::startup,
                TerrainHeightMaps::startup,
                TerrainChunkView::setup,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                TerrainChunkView::update_meshes,
                TerrainChunk::update_chunk_bounds,
                TerrainCollider::update_terrain_collider,
                HeightMapUniforms::update_from_settings,
                // TerrainGpuImages::update_textures_from_gpu,
                TerrainChunkView::update_debug_quad_tree,
                startup_terrain,
            )
                .chain(),
        )
        .add_observer(update_materials_for_tree);

        app.add_plugins((
            ExtractResourcePlugin::<TerrainGpuImages>::default(),
            ExtractResourcePlugin::<HeightMapUniforms>::default(),
            ExtractComponentPlugin::<TerrainChunk>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, TerrainPipeline::init_pipeline)
            .add_systems(
                Render,
                (TerrainBindGroups::prepare).in_set(RenderSystems::PrepareBindGroups),
            );
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(TerrainRenderLabel, TerrainRenderNode::default());
        render_graph.add_node_edge(TerrainRenderLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Resource, Debug)]
pub struct TerrainSettings {
    pub frequency: f32,
    pub lacunarity: f32,
    pub octaves: f32,
    pub persistence: f32,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            frequency: 0.3,
            lacunarity: 1.4,
            octaves: 11.,
            persistence: 0.6,
            x_bounds: (-5., 5.),
            y_bounds: (-5., 5.),
        }
    }
}

fn generate_stitched_plane_northeast(
    full_x: usize,
    full_z: usize,
    low_x: usize,
    low_z: usize,
    width: f32,
    depth: f32,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let x_step = width / (full_x - 1) as f32;
    let z_step = depth / (full_z - 1) as f32;

    // ===============================================================
    // 1. Generate full-resolution interior grid EXCEPT:
    //    - last row (north)
    //    - last column (east)
    // ===============================================================

    for z in 0..(full_z - 1) {
        let z_pos = z as f32 * z_step - depth * 0.5;
        for x in 0..(full_x - 1) {
            let x_pos = x as f32 * x_step - width * 0.5;

            positions.push([x_pos, 0.0, z_pos]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([
                x as f32 / (full_x - 1) as f32,
                z as f32 / (full_z - 1) as f32,
            ]);
        }
    }

    let interior_width = full_x - 1;

    // ===============================================================
    // 2. Low-res north row (full width, but low_x samples)
    // ===============================================================

    let z_north = depth * 0.5;
    for i in 0..low_x {
        let t = i as f32 / (low_x - 1) as f32;
        let x_pos = t * width - width * 0.5;

        positions.push([x_pos, 0.0, z_north]);
        normals.push([0.0, 1.0, 0.0]);

        uvs.push([t, 1.0]);
    }

    let north_start = positions.len() - low_x;

    // ===============================================================
    // 3. Low-res east column (full height, but low_z samples)
    // ===============================================================

    let x_east = width * 0.5;
    for i in 0..low_z {
        let t = i as f32 / (low_z - 1) as f32;
        let z_pos = t * depth - depth * 0.5;

        positions.push([x_east, 0.0, z_pos]);
        normals.push([0.0, 1.0, 0.0]);

        uvs.push([1.0, t]);
    }

    let east_start = positions.len() - low_z;

    // ===============================================================
    // 4. Interior full-resolution quads
    // ===============================================================

    let mut indices = Vec::new();

    for z in 0..(full_z - 2) {
        for x in 0..(full_x - 2) {
            let i0 = (z * interior_width + x) as u32;
            let i1 = i0 + 1;
            let i2 = (z + 1) as u32 * interior_width as u32 + x as u32;
            let i3 = i2 + 1;

            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // ===============================================================
    // 5. Stitch NORTH edge
    // ===============================================================

    let last_interior_row_start = (full_z - 2) * interior_width;

    for i in 0..(low_x - 1) {
        let lo0 = (north_start + i) as u32;
        let lo1 = lo0 + 1;

        let hi0 = (last_interior_row_start + i * (interior_width - 1) / (low_x - 1)) as u32;
        let hi1 = hi0 + 1;
        let hi2 = hi0 + 2;

        indices.extend_from_slice(&[hi0, lo0, hi1, hi1, lo0, lo1, hi1, lo1, hi2]);
    }

    // ===============================================================
    // 6. Stitch EAST edge
    // ===============================================================

    for i in 0..(low_z - 1) {
        let lo0 = (east_start + i) as u32;
        let lo1 = lo0 + 1;

        let z0 = i * (full_z - 1) / (low_z - 1);
        let hi0 = (z0 * interior_width + (interior_width - 1)) as u32;
        let hi1 = hi0 + interior_width as u32;
        let hi2 = hi1 + interior_width as u32;

        indices.extend_from_slice(&[hi0, hi1, lo0, lo0, hi1, lo1, hi1, hi2, lo1]);
    }

    // ===============================================================
    // 7. Build mesh
    // ===============================================================

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn generate_stitched_plane_west(
    full_z: usize,
    low_z: usize,
    x_count: usize,
    width: f32,
    depth: f32,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let x_step = width / (x_count - 1) as f32;
    let z_step = depth / (full_z - 1) as f32;

    // ============================================================
    // 1. Generate full-resolution columns EXCEPT the first column
    //    (just like your north version skips the last row)
    // ============================================================

    for z in 0..full_z {
        let z_pos = z as f32 * z_step - depth * 0.5;

        // skip x = 0 (west side is low-res version)
        for x in 1..x_count {
            let x_pos = x as f32 * x_step - width * 0.5;

            positions.push([x_pos, 0.0, z_pos]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([
                x as f32 / (x_count - 1) as f32,
                z as f32 / (full_z - 1) as f32,
            ]);
        }
    }

    let interior_width = x_count - 1; // number of columns we generated

    // ============================================================
    // 2. Generate low-res WEST column (x = -width/2)
    // ============================================================

    let x_west = -width * 0.5;

    for z in 0..low_z {
        let t = z as f32 / (low_z - 1) as f32;
        let z_pos = t * depth - depth * 0.5;

        positions.push([x_west, 0.0, z_pos]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.0, t]);
    }

    let west_start = positions.len() - low_z;

    // ============================================================
    // 3. Full-resolution interior quads
    // ============================================================

    let mut indices = Vec::new();

    for z in 0..(full_z - 1) {
        for x in 0..(interior_width - 1) {
            let i0 = (z * interior_width + x) as u32;
            let i1 = i0 + 1;
            let i2 = (z + 1) as u32 * interior_width as u32 + x as u32;
            let i3 = i2 + 1;

            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // ============================================================
    // 4. WEST stitch (low-res left + full-res interior)
    // ============================================================

    for i in 0..(low_z - 1) {
        let lo0 = (west_start + i) as u32;
        let lo1 = lo0 + 1;

        // Map low-res z to full-res z
        let hi_z0 = i * (full_z - 1) / (low_z - 1);
        let hi0 = (hi_z0 * interior_width) as u32; // left interior vertex
        let hi1 = hi0 + 1; // interior next x
        let hi2 = hi0 + interior_width as u32; // interior next z row

        indices.extend_from_slice(&[hi0, lo0, hi1, hi1, lo0, lo1, hi1, lo1, hi2]);
    }

    // ============================================================
    // 5. Build mesh
    // ============================================================

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn generate_stitched_plane_east(
    full_z: usize,
    low_z: usize,
    x_count: usize,
    width: f32,
    depth: f32,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let x_step = width / (x_count - 1) as f32;
    let z_step = depth / (full_z - 1) as f32;

    // --- 1. Generate full-resolution columns except east edge ---
    for z in 0..full_z {
        let z_pos = z as f32 * z_step - depth * 0.5;
        for x in 0..(x_count - 1) {
            let x_pos = x as f32 * x_step - width * 0.5;
            positions.push([x_pos, 0.0, z_pos]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([
                x as f32 / (x_count - 1) as f32,
                z as f32 / (full_z - 1) as f32,
            ]);
        }
    }

    // --- 2. Low-res east column ---
    let x_last = width * 0.5;
    for z in 0..low_z {
        let t = z as f32 / (low_z - 1) as f32;
        let z_pos = t * depth - depth * 0.5;
        positions.push([x_last, 0.0, z_pos]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([1.0, t]);
    }

    let mut indices = Vec::new();

    // --- 3. Full-res indices ---
    for z in 0..(full_z - 1) {
        for x in 0..(x_count - 2) {
            let i0 = (z * (x_count - 1) + x) as u32;
            let i1 = i0 + 1;
            let i2 = ((z + 1) * (x_count - 1) + x) as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // --- 4. Stitch ---
    let full_col_count = (x_count - 1);
    let east_low_start = positions.len() - low_z;

    for lz in 0..(low_z - 1) {
        let lo0 = east_low_start + lz;
        let lo1 = lo0 + 1;

        let hi0 = lz * 2 * full_col_count + (full_col_count - 1);
        let hi1 = hi0 + full_col_count;
        let hi2 = hi1 + full_col_count;

        indices.extend_from_slice(&[
            hi0 as u32, hi1 as u32, lo0 as u32, lo0 as u32, hi1 as u32, lo1 as u32, hi1 as u32,
            hi2 as u32, lo1 as u32,
        ]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn generate_stitched_plane_north(
    full_x: usize,
    low_x: usize,
    z_count: usize,
    width: f32,
    depth: f32,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let x_step = width / (full_x - 1) as f32;
    let z_step = depth / (z_count - 1) as f32;

    // --- 1. Generate full-resolution rows ---
    for z in 0..(z_count - 1) {
        let z_pos = z as f32 * z_step - depth * 0.5;
        for x in 0..full_x {
            let x_pos = x as f32 * x_step - width * 0.5;
            positions.push([x_pos, 0.0, z_pos]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([
                x as f32 / (full_x - 1) as f32,
                z as f32 / (z_count - 1) as f32,
            ]);
        }
    }

    // --- 2. Low-res final row ---
    let z_last = depth * 0.5;
    for x in 0..low_x {
        let t = x as f32 / (low_x - 1) as f32;
        let x_pos = t * width - width * 0.5;
        positions.push([x_pos, 0.0, z_last]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([t, 1.0]);
    }

    // --- 3. Indices for full-res rows ---
    let mut indices = Vec::new();
    for z in 0..(z_count - 2) {
        for x in 0..(full_x - 1) {
            let i0 = (z * full_x + x) as u32;
            let i1 = i0 + 1;
            let i2 = ((z + 1) * full_x + x) as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // --- 4. Stitch ---
    let last_full_row_start = (z_count - 2) * full_x;
    let low_row_start = positions.len() - low_x;

    for lx in 0..(low_x - 1) {
        let lo0 = low_row_start + lx;
        let lo1 = lo0 + 1;

        let hi0 = last_full_row_start + lx * 2;
        let hi1 = hi0 + 1;
        let hi2 = hi0 + 2;

        indices.extend_from_slice(&[
            hi0 as u32, lo0 as u32, hi1 as u32, hi1 as u32, lo0 as u32, lo1 as u32, hi1 as u32,
            lo1 as u32, hi2 as u32,
        ]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn generate_stitched_plane_south(
    full_x: usize,
    low_x: usize,
    z_count: usize,
    width: f32,
    depth: f32,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let x_step = width / (full_x - 1) as f32;
    let z_step = depth / (z_count - 1) as f32;

    // --- 1. Low-res first row alone ---
    let z_first = -depth * 0.5;
    for x in 0..low_x {
        let t = x as f32 / (low_x - 1) as f32;
        let x_pos = t * width - width * 0.5;
        positions.push([x_pos, 0.0, z_first]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([t, 0.0]);
    }

    // --- 2. Full-resolution rows ---
    for z in 1..z_count {
        let z_pos = z as f32 * z_step - depth * 0.5;
        for x in 0..full_x {
            let x_pos = x as f32 * x_step - width * 0.5;
            positions.push([x_pos, 0.0, z_pos]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([
                x as f32 / (full_x - 1) as f32,
                z as f32 / (z_count - 1) as f32,
            ]);
        }
    }

    let full_rows_start = low_x;
    let mut indices = Vec::new();

    // --- 3. Full-resolution quad rows ---
    for z in 0..(z_count - 2) {
        for x in 0..(full_x - 1) {
            let row0 = full_rows_start + z * full_x;
            let row1 = full_rows_start + (z + 1) * full_x;
            let i0 = (row0 + x) as u32;
            let i1 = i0 + 1;
            let i2 = (row1 + x) as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    // --- 4. Stitch low-res first row to first full-res row ---
    let low_row = 0;
    let first_full_row = full_rows_start;

    for lx in 0..(low_x - 1) {
        let lo0 = low_row + lx;
        let lo1 = lo0 + 1;

        let hi0 = first_full_row + lx * 2;
        let hi1 = hi0 + 1;
        let hi2 = hi0 + 2;

        indices.extend_from_slice(&[
            lo0 as u32, hi0 as u32, hi1 as u32, lo0 as u32, hi1 as u32, lo1 as u32, lo1 as u32,
            hi1 as u32, hi2 as u32,
        ]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum StitchIn {
    None,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl StitchIn {
    fn from_neighbour_in(neighbour_ins: Vec<Neighbouring>) -> Self {
        match neighbour_ins.as_slice() {
            [Neighbouring::Top, Neighbouring::Left] => Self::NorthWest,
            [Neighbouring::Top] => Self::North,
            [Neighbouring::Top, Neighbouring::Right] => Self::NorthEast,
            [Neighbouring::Right] => Self::East,
            [Neighbouring::Right, Neighbouring::Bottom] => Self::SouthEast,
            [Neighbouring::Bottom] => Self::South,
            [Neighbouring::Bottom, Neighbouring::Left] => Self::SouthWest,
            [Neighbouring::Left] => Self::West,
            _ => Self::None,
        }
    }

    fn to_color(&self) -> Color {
        match self {
            StitchIn::None => WHITE,
            StitchIn::North => RED,
            StitchIn::NorthEast => YELLOW,
            StitchIn::East => GREEN,
            StitchIn::SouthEast => ORANGE,
            StitchIn::South => BLUE,
            StitchIn::SouthWest => TEAL,
            StitchIn::West => PURPLE,
            StitchIn::NorthWest => VIOLET,
        }
        .into()
    }
}

#[derive(Debug)]
enum QuadNode {
    Branch {
        children: Box<[QuadNode; 4]>,
        chunk: QuadChunk,
    },
    Leaf(QuadChunk),
}

#[derive(Debug)]
struct QuadChunk {
    bound: Aabb2d,
    level: usize,
    id: u32,
}

impl QuadChunk {
    fn new(bounds: Aabb2d, level: usize) -> Self {
        let id = Self::chunk_id(&bounds);
        Self {
            bound: bounds,
            level,
            id,
        }
    }

    fn split(aabb: Aabb2d, level: usize) -> [Self; 4] {
        QuadNode::split_bounds(aabb).map(|bounds| Self::new(bounds, level))
    }

    // Using the minimum chunk size to get our coordinates?!
    fn pos(&self, chunk_size: Vec2) -> IVec2 {
        let min = self.bound.min;
        let ix = (min.x / chunk_size.x).floor() as i32;
        let iy = (min.y / chunk_size.y).floor() as i32;
        IVec2::new(ix, iy)
    }

    // todo: Chatgpt things this should be a morton code :shrug:
    fn chunk_id(bounds: &Aabb2d) -> u32 {
        let center = bounds.center();
        let ix = center.x.floor() as i16;
        let iy = center.y.floor() as i16;
        ((ix as i32) << 16 | iy as i32) as u32
    }
}

impl QuadNode {
    fn chunk(&self) -> &QuadChunk {
        match self {
            QuadNode::Branch { children: _, chunk } => chunk,
            QuadNode::Leaf(quad_chunk) => quad_chunk,
        }
    }
    fn debug_chunks(&self, point: Vec2, gizmos: &mut Gizmos) {
        for (_id, node) in &self.nodes_with_id(point) {
            let chunk = node.chunk();
            gizmos.primitive_3d(
                &Cuboid::new(
                    chunk.bound.half_size().x * 2.,
                    0.,
                    chunk.bound.half_size().y * 2.,
                ),
                Isometry3d::from_translation(
                    chunk
                        .bound
                        .center()
                        .with_y(1.)
                        .extend(chunk.bound.center().y),
                ),
                BLUE,
            );
        }
        for (_id, node) in &self.refined_nodes_with_id(point) {
            let chunk = node.chunk();
            gizmos.primitive_3d(
                &Cuboid::new(
                    chunk.bound.half_size().x * 2.,
                    0.,
                    chunk.bound.half_size().y * 2.,
                ),
                Isometry3d::from_translation(
                    chunk
                        .bound
                        .center()
                        .with_y(0.)
                        .extend(chunk.bound.center().y),
                ),
                RED,
            );
        }
    }

    /// Returns all chunks
    fn all(&self) -> Vec<&QuadNode> {
        let mut nodes = Vec::with_capacity(4);
        match self {
            QuadNode::Branch { children, chunk: _ } => {
                nodes.push(self);
                let child_array: &[QuadNode; 4] = children;
                for child in child_array {
                    nodes.extend(child.all());
                }
            }
            QuadNode::Leaf(_) => nodes.push(self),
        };
        nodes
    }

    // step one is to find the chunk our point is in
    fn refine<'a>(point: Vec2, nodes: &HashMap<u32, &'a QuadNode>) -> HashMap<u32, &'a QuadNode> {
        let Some(start_node) = nodes.values().find(|node| {
            let bound = node.chunk().bound;
            let delta = point - bound.center();

            delta.x.abs() <= bound.half_size().x && delta.y.abs() <= bound.half_size().y
        }) else {
            return nodes.clone();
        };
        let mut new_nodes = HashMap::new();
        let mut queue: VecDeque<&QuadNode> = VecDeque::new();
        let mut seen: HashSet<u32> = HashSet::new();
        seen.insert(start_node.chunk().id);
        queue.push_back(start_node);
        while let Some(node) = queue.pop_front() {
            let chunk = node.chunk();
            let neighbours = node.neighbours(&nodes);
            let mut should_refine = false;
            for (_direction, neighbour_node) in &neighbours {
                let neighbour_chunk = neighbour_node.chunk();
                // do we refine our current chunk?
                if chunk.level as i32 - neighbour_chunk.level as i32 > 1 {
                    should_refine = true;
                }
                if seen.insert(neighbour_chunk.id) {
                    queue.push_back(neighbour_node);
                }
            }
            if should_refine {
                match node {
                    QuadNode::Branch { children, chunk: _ } => {
                        let child_array: &[QuadNode; 4] = children;
                        for child in child_array {
                            new_nodes.insert(child.chunk().id, child);
                        }
                    }
                    QuadNode::Leaf(quad_chunk) => {
                        // we can't refine this chunk anymore, so when we get to the neighbour
                        // we will refine it instead.
                        new_nodes.insert(quad_chunk.id, node);
                    }
                }
            } else {
                new_nodes.insert(chunk.id, node);
            }
        }
        return new_nodes;
    }

    /// Returns the direction *from r1 to r2* if they border each other.
    /// Returns None if they do not touch or only touch at a corner.
    pub fn bordering_direction(b1: &Aabb2d, b2: &Aabb2d) -> Option<Neighbouring> {
        // Convenience
        let r1_left = b1.min.x;
        let r1_right = b1.max.x;
        let r1_bottom = b1.min.y;
        let r1_top = b1.max.y;

        let r2_left = b2.min.x;
        let r2_right = b2.max.x;
        let r2_bottom = b2.min.y;
        let r2_top = b2.max.y;

        // --- Shared edge overlap helpers ---
        let overlap_x = r1_left < r2_right && r1_right > r2_left;
        let overlap_y = r1_bottom < r2_top && r1_top > r2_bottom;

        // --- Check top adjacency (r2 is directly above r1) ---
        if (r1_top == r2_bottom) && overlap_x {
            return Some(Neighbouring::Top);
        }

        // --- Check bottom adjacency (r2 is directly below r1) ---
        if (r1_bottom == r2_top) && overlap_x {
            return Some(Neighbouring::Bottom);
        }

        // --- Check right adjacency (r2 is directly right of r1) ---
        if (r1_right == r2_left) && overlap_y {
            return Some(Neighbouring::Right);
        }

        // --- Check left adjacency (r2 is directly left of r1) ---
        if (r1_left == r2_right) && overlap_y {
            return Some(Neighbouring::Left);
        }

        None
    }

    // annoyingly we have to go trhough every chunk in the
    fn neighbours<'a>(
        &self,
        nodes: &HashMap<u32, &'a QuadNode>,
    ) -> Vec<(Neighbouring, &'a QuadNode)> {
        let mut neighbours: Vec<(Neighbouring, &'a QuadNode)> = Vec::with_capacity(4);
        let chunk = self.chunk();
        for (_key, neighbour_node) in nodes {
            let neighbour_chunk = neighbour_node.chunk();
            if let Some(neighbouring) =
                Self::bordering_direction(&chunk.bound, &neighbour_chunk.bound)
            {
                neighbours.push((neighbouring, neighbour_node));
            }
        }
        neighbours
    }

    /// Returns a list of the nodes to be rendered
    /// we could return an Option, this way we can add entities later
    fn nodes_with_id(&self, point: Vec2) -> HashMap<u32, &QuadNode> {
        let mut nodes = HashMap::default();
        // note: For the subdivide check we currently just get the diagonal size, if we're within
        // that distance from the quad's center we subdivide
        match self {
            QuadNode::Branch { children, chunk }
                if (chunk.bound.center().distance_squared(point) as u32)
                    < (((chunk.bound.half_size().x + chunk.bound.half_size().y) * 1.25) as u32)
                        .pow(2) =>
            {
                let child_array: &[QuadNode; 4] = children;
                for child in child_array {
                    nodes.extend(child.nodes_with_id(point));
                }
            }
            QuadNode::Branch { children: _, chunk } => {
                nodes.insert(chunk.id, self);
            }
            QuadNode::Leaf(chunk) => {
                nodes.insert(chunk.id, self);
            }
        };
        nodes
    }

    // todo: We need to encode which directions we need to stitch in
    fn refined_nodes_with_id(&self, point: Vec2) -> HashMap<u32, &QuadNode> {
        let mut nodes = self.nodes_with_id(point);
        loop {
            let refined_nodes = Self::refine(point, &nodes);
            if nodes.len() == refined_nodes.len() {
                return refined_nodes;
            }
            nodes = refined_nodes;
        }
    }

    fn split_bounds(bounds: Aabb2d) -> [Aabb2d; 4] {
        let quarter_size = bounds.half_size() / 2.;
        let center = bounds.center();
        let bot_left = Aabb2d::new(center - quarter_size, quarter_size);
        let top_right = Aabb2d::new(center + quarter_size, quarter_size);
        let top_left = Aabb2d::new(
            Vec2::new(center.x - quarter_size.x, center.y + quarter_size.y),
            quarter_size,
        );
        let bot_right = Aabb2d::new(
            Vec2::new(center.x + quarter_size.x, center.y - quarter_size.y),
            quarter_size,
        );
        [top_left, top_right, bot_left, bot_right]
    }

    fn new(bounds: Aabb2d, depth: usize) -> Self {
        if depth == 1 {
            todo!("return a single root node");
        }

        fn inner(bounds: Aabb2d, depth: usize, max_depth: usize) -> QuadNode {
            if depth == 1 {
                let nodes = QuadChunk::split(bounds, 0).map(|chunk| QuadNode::Leaf(chunk));
                return QuadNode::Branch {
                    children: Box::new(nodes),
                    chunk: QuadChunk::new(bounds, 1),
                };
            }

            let nodes =
                QuadNode::split_bounds(bounds).map(|bounds| inner(bounds, depth - 1, max_depth));
            return QuadNode::Branch {
                children: Box::new(nodes),
                chunk: QuadChunk::new(bounds, depth),
            };
        }
        inner(bounds, depth, depth)
    }
}

enum TerrainRenderState {
    /// Ensure init scripts are loaded
    Loading,
    /// Updates a height map at (pos, buffer_index)
    Update,
    Coarse,
    // Runs vertex height adjustment and fragment shaders
    Monitor,
}
struct TerrainRenderNode {
    state: TerrainRenderState,
}

impl Default for TerrainRenderNode {
    fn default() -> Self {
        Self {
            state: TerrainRenderState::Loading,
        }
    }
}
impl render_graph::Node for TerrainRenderNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<TerrainPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        match &self.state {
            TerrainRenderState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.height_map_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = TerrainRenderState::Update;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets\n{err}")
                    }
                    _ => {}
                }
            }
            TerrainRenderState::Coarse => {}
            TerrainRenderState::Update => {
                if !world.is_resource_changed::<HeightMapUniforms>() {
                    self.state = TerrainRenderState::Monitor;
                }
            }
            TerrainRenderState::Monitor => {
                if world.is_resource_changed::<HeightMapUniforms>() {
                    self.state = TerrainRenderState::Update;
                }
            }
        }
    }
    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        world: &'w World,
    ) -> std::result::Result<(), render_graph::NodeRunError> {
        let height_map_params = &world.resource::<HeightMapUniforms>();
        let terrain_images = world.resource::<TerrainGpuImages>();
        let terrain_bind_groups = world.resource::<TerrainBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let terrain_pipeline = world.resource::<TerrainPipeline>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let queue = world.resource::<RenderQueue>();

        match &self.state {
            TerrainRenderState::Loading => {}
            TerrainRenderState::Coarse => {}
            TerrainRenderState::Update => {
                let height_map_gpu = gpu_images.get(&terrain_images.height_map).unwrap();
                let size = height_map_gpu.size_2d();

                let pipeline = pipeline_cache
                    .get_compute_pipeline(terrain_pipeline.height_map_pipeline)
                    .unwrap();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_bind_group(0, &terrain_bind_groups.height_map, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(size.x / 8, size.y / 8, 1);
            }
            TerrainRenderState::Monitor => {}
        }
        Ok(())
    }
}

#[derive(Resource, Default)]
struct TerrainLoadingImages {
    texture_loaded: bool,
    ground: Handle<Image>,
    ground_normal: Handle<Image>,

    slope: Handle<Image>,
    slope_normal: Handle<Image>,
}

impl TerrainLoadingImages {
    fn is_loaded(&self, asset_server: &AssetServer) -> bool {
        asset_server.is_loaded(&self.ground)
            && asset_server.is_loaded(&self.ground_normal)
            && asset_server.is_loaded(&self.slope)
            && asset_server.is_loaded(&self.slope_normal)
    }
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct HeightMapUniforms {
    frequency: f32,
    lacunarity: f32,
    octaves: i32,
    persistence: f32,
    x_from: f32,
    x_to: f32,
    y_from: f32,
    y_to: f32,
}

impl HeightMapUniforms {
    fn update_from_settings(
        settings: Res<TerrainSettings>,
        mut uniforms: ResMut<HeightMapUniforms>,
    ) {
        if !settings.is_changed() {
            return;
        }
        uniforms.lacunarity = settings.lacunarity;
        uniforms.persistence = settings.persistence;
        uniforms.frequency = settings.frequency;
        uniforms.octaves = settings.octaves as i32;
    }
}

#[derive(Resource)]
struct TerrainPipeline {
    height_map_bind_group_layout: BindGroupLayout,
    height_map_pipeline: CachedComputePipelineId,
}

impl TerrainPipeline {
    fn init_pipeline(
        mut commands: Commands,
        render_device: Res<RenderDevice>,
        asset_server: Res<AssetServer>,
        pipeline_cache: Res<PipelineCache>,
    ) {
        // Create bind group and pipeline
        let bind_group_layout = render_device.create_bind_group_layout(
            "compute_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<HeightMapUniforms>(false),
                ),
            ),
        );

        let shader: Handle<Shader> = asset_server.load("shaders/height_map.wgsl");
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("perlin_compute_pipeline")),
            layout: vec![bind_group_layout.clone()],
            shader,
            entry_point: Some(Cow::from("main")),
            ..Default::default()
        });
        commands.insert_resource(TerrainPipeline {
            height_map_bind_group_layout: bind_group_layout,
            height_map_pipeline: pipeline,
        });
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct TerrainBindGroups {
    height_map: BindGroup,
}

impl TerrainBindGroups {
    fn prepare(
        mut commands: Commands,
        terrain_pipeline: Res<TerrainPipeline>,
        gpu_images: Res<RenderAssets<GpuImage>>,
        terrain_images: Res<TerrainGpuImages>,
        height_map_uniforms: Res<HeightMapUniforms>,
        render_device: Res<RenderDevice>,
        queue: Res<RenderQueue>,
    ) {
        let height_map_gpu = gpu_images.get(&terrain_images.height_map).unwrap();
        let _size = height_map_gpu.size_2d();
        let mut uniform_buffer = UniformBuffer::from(height_map_uniforms.into_inner());
        uniform_buffer.write_buffer(&render_device, &queue);

        let bind_group = render_device.create_bind_group(
            Some("height_map_bind_group"),
            // todo(next-bevy): Will use pipeline cache for this
            &terrain_pipeline.height_map_bind_group_layout,
            &BindGroupEntries::sequential((&height_map_gpu.texture_view, &uniform_buffer)),
        );
        commands.insert_resource(TerrainBindGroups {
            height_map: bind_group,
        });
    }
}

// these textures get uploaded to the GPU
#[derive(Resource, Clone, ExtractResource)]
struct TerrainGpuImages {
    /// This is our global height maps for our quad trees,
    /// when a new quad tree is created we generate the height map once and store it.
    /// I GUESS!?
    height_map: Handle<Image>,
    cpu_update_timer: Option<Timer>,
}

// these textures get uploaded to the GPU
#[derive(Component)]
struct Tree;

fn startup_test_tree(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Start loading the asset as a scene and store a reference to it in a
    // SceneRoot component. This component will automatically spawn a scene
    // containing our mesh once it has loaded.
    let handle =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/nature/tree_04.glb"));
    let mesh_scene = SceneRoot(handle);
    commands.spawn((mesh_scene, Tree, Transform::from_xyz(2., 0., 2.)));
}

fn update_materials_for_tree(
    scene_ready: On<SceneInstanceReady>,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    tree_q: Query<(), With<Tree>>,
    children: Query<&Children>,
    mut mesh_materials: Query<(&mut MeshMaterial3d<StandardMaterial>, &GltfMeshName)>,
) {
    if tree_q.get(scene_ready.entity).is_err() {
        return;
    }
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok((ref mut id, mesh_name)) = mesh_materials.get_mut(descendant) else {
            continue;
        };
        match mesh_name.0.as_str() {
            "leaves" => {
                id.0 = materials.add(StandardMaterial {
                    base_color: Color::srgb_u8(102, 153, 51),
                    base_color_texture: Some(asset_server.load_with_settings(
                        "models/nature/F1_Leaves_Map1.png",
                        move |s: &mut ImageLoaderSettings| {
                            let sampler_desc = ImageSamplerDescriptor {
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..Default::default()
                            };
                            s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
                        },
                    )),
                    alpha_mode: AlphaMode::Blend,
                    ..Default::default()
                });
            }
            "trunk" => {
                id.0 = materials.add(StandardMaterial {
                    base_color_texture: Some(asset_server.load_with_settings(
                        "models/nature/bark1_baseColor.png",
                        move |s: &mut ImageLoaderSettings| {
                            let sampler_desc = ImageSamplerDescriptor {
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..Default::default()
                            };
                            s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
                        },
                    )),
                    normal_map_texture: Some(asset_server.load_with_settings(
                        "models/nature/FBX_Textures/FBX/bark1_normal_GL.png",
                        move |s: &mut ImageLoaderSettings| {
                            let sampler_desc = ImageSamplerDescriptor {
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..Default::default()
                            };
                            s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
                        },
                    )),
                    ..Default::default()
                });
            }
            _ => (),
        }
    }
}

impl TerrainGpuImages {
    // my idea is we monitor for a TerrainImagesEvent that signals a change has occured, when we
    // see that a change has occured we wait a few frames until the change stops then we trigger a
    // Readback of the texture, NEATO.
    // todo(minor): Could implement a despawning of all the other readbacks when we start this
    fn update_textures_from_gpu(
        mut commands: Commands,
        mut terrain_images: ResMut<TerrainGpuImages>,
        height_maps: Res<TerrainHeightMaps>,
        height_map_uniforms: Res<HeightMapUniforms>,
        time: Res<Time>,
    ) {
        if height_map_uniforms.is_changed() {
            terrain_images.cpu_update_timer = Some(Timer::new(
                Duration::new(HEIGHTMAP_DEBOUNCE_SECS, 0),
                TimerMode::Once,
            ));
        }
        if let Some(ref mut timer) = terrain_images.cpu_update_timer {
            timer.tick(time.delta());
            if timer.is_finished() {
                terrain_images.cpu_update_timer = None;
                let lod0_handle = height_maps.height_map.clone();
                commands
                    .spawn(Readback::texture(terrain_images.height_map.clone()))
                    .observe(
                        move |ev: On<ReadbackComplete>,
                              mut commands: Commands,
                              mut terrain_height_maps: ResMut<TerrainHeightMaps>,
                              mut assets: ResMut<Assets<Image>>| {
                            if let Some(image) = assets.get_mut(&lod0_handle) {
                                image.data = Some(ev.data.clone());
                                terrain_height_maps.set_changed();
                            } else {
                                panic!("cpu image for height map not found!");
                            }
                            commands.entity(ev.entity).despawn();
                        },
                    );
            }
        }
    }

    fn startup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        terrain_settings: Res<TerrainSettings>,
    ) {
        let mut height_map_image =
            Image::new_target_texture(HEIGHT_MAP_SIZE, HEIGHT_MAP_SIZE, TextureFormat::Rgba32Float);
        height_map_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        height_map_image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC;
        // height_map.sampler = ImageSampler::nearest();

        commands.insert_resource(TerrainGpuImages {
            height_map: images.add(height_map_image.clone()),
            cpu_update_timer: Some(Timer::new(
                Duration::new(HEIGHTMAP_DEBOUNCE_SECS, 0),
                TimerMode::Once,
            )),
        });
        // we need to calculate the overlap
        commands.insert_resource(HeightMapUniforms {
            frequency: terrain_settings.frequency,
            lacunarity: terrain_settings.lacunarity,
            octaves: terrain_settings.octaves as i32,
            persistence: terrain_settings.persistence,
            x_from: -1.,
            x_to: 1.,
            y_from: -1.,
            y_to: 1.,
        });
    }
}

// these heightmaps exist only on the CPU and are read back from the GPU on change.
// these height maps only use the R channel compared to the GPU which stores the derivatives as
// well.
#[derive(Resource, Clone)]
struct TerrainHeightMaps {
    height_map: Handle<Image>,
}

impl TerrainHeightMaps {
    fn startup(mut commands: Commands, mut assets: ResMut<Assets<Image>>) {
        let mut height_map =
            Image::new_target_texture(HEIGHT_MAP_SIZE, HEIGHT_MAP_SIZE, TextureFormat::Rgba32Float);
        height_map.asset_usage = RenderAssetUsages::MAIN_WORLD;
        commands.insert_resource(TerrainHeightMaps {
            height_map: assets.add(height_map),
        });
    }
}

#[derive(Component)]
struct TerrainCollider {
    height_map: Handle<Image>,
    mesh: Handle<Mesh>,
    index: u32,
}

impl TerrainCollider {
    fn update_terrain_collider(
        mut commands: Commands,
        terrain_collider_q: Query<(Entity, &TerrainCollider)>,
        terrain_height_maps: Res<TerrainHeightMaps>,
        images: Res<Assets<Image>>,
        meshes: Res<Assets<Mesh>>,
    ) {
        if !terrain_height_maps.is_changed() {
            return;
        }
        for (entity, terrain_collider) in terrain_collider_q {
            if let (Some(image), Some(mesh)) = (
                images.get(&terrain_collider.height_map),
                meshes.get(&terrain_collider.mesh),
            ) {
                let collider =
                    Self::collider_from_image_and_mesh(image, terrain_collider.index, mesh);
                commands
                    .entity(entity)
                    .insert((RigidBody::Static, collider));
            }
        }
    }

    fn collider_from_image_and_mesh(image: &Image, layer: u32, mesh: &Mesh) -> Collider {
        let image_size = image.size();
        let mut mesh = mesh.clone();
        let half = CHUNK_HALF_SIZE;
        let full = CHUNK_HALF_SIZE * 2.;
        if let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
        {
            for vertex in positions.iter_mut() {
                // todo: Why does this make it align properly? :confused:
                let x = vertex[0] + 1.;
                let z = vertex[2] + 1.;
                let u = (x + half) / full;
                let v = (z + half) / full;
                // todo: This is actually incorrect and I'm not sure why, it only works when the
                // chunk size == our image size, womp womp
                let p_x = (u * (image_size.x as f32 - 1.)).round() as u32;
                let p_y = (v * (image_size.y as f32 - 1.)).round() as u32;
                let value = image
                    .get_color_at_3d(p_x, p_y, layer)
                    .expect(&format!("failed to get coordinates for ({},{})", p_x, p_y))
                    .to_linear();
                vertex[1] += value.red * TERRAIN_MAX_HEIGHT;
            }
        }
        Collider::trimesh_from_mesh(&mesh).unwrap()
    }
}

#[derive(Component, Clone, Debug, ExtractComponent)]
struct TerrainChunk {
    pub center: Vec2,
    pub pos: IVec2,
    pub size: IVec2,
    pub heights_buffer: Handle<ShaderStorageBuffer>,
}

impl TerrainChunk {
    // matches the height of the terrain to the height the vertex shader puts out.
    fn update_chunk_bounds(mut q: Query<&mut Aabb, (Added<Aabb>, With<TerrainChunk>)>) {
        for mut aabb in &mut q {
            // todo: Height need's to be passed into the vertex shader
            let height = TERRAIN_MAX_HEIGHT; // max y
            aabb.half_extents.y = height;
        }
    }
}

// marker component that will load chunks around this entity
// the chunk cam could also keep a list of entities that it's spawned,
// we can then use that list to check if we need to despawn or change the eleemnt
#[derive(Component)]
#[require(Transform)]
pub struct TerrainChunkView {
    // a problem with this is what do we do when we're close to the bounds of the map? a fix is to
    // basically have another "clip map" that monitors when we're near the edge and causes the
    // entire quad tree to be recalculated over the top of us. the problem here though is that we'd
    // cause a lag spike, a decent solution would be to have a double buffer and load the next quad
    // tree in the background swapping over after some time.
    quad: QuadNode,
    terrain_size: f32,
    terrain_height: f32,
    chunk_size: f32,
}

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Neighbouring {
    Top,
    Right,
    Bottom,
    Left,
}

impl TerrainChunkView {
    fn update_debug_quad_tree(
        chunk_views: Query<(&GlobalTransform, &TerrainChunkView), Changed<Transform>>,
        mut gizmos: Gizmos,
    ) {
        for (g_transform, view) in chunk_views {
            view.quad.debug_chunks(
                g_transform
                    .translation()
                    .truncate()
                    .with_y(g_transform.translation().z),
                &mut gizmos,
            );
        }
    }

    // todo: The meshes are created with a fixed size. we need to create the meshes in our setup
    // function to be the size of a chunk.
    pub fn new(chunk_size: f32, world_size: f32) -> Self {
        let depth = (world_size / chunk_size).log2() as usize;
        Self {
            quad: QuadNode::new(
                Aabb2d {
                    min: Vec2::splat(0.),
                    max: Vec2::splat(world_size),
                },
                depth,
            ),
            terrain_height: TERRAIN_MAX_HEIGHT,
            terrain_size: world_size,
            chunk_size: chunk_size,
        }
    }

    /// Updates meshes based on the distance to the camera,
    fn setup(
        mut commands: Commands,
        mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
        terrain_loading: Res<TerrainLoadingImages>,
        terrain_images: Res<TerrainGpuImages>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut height_map_uniforms: ResMut<HeightMapUniforms>,
        chunk_views: Query<(Entity, &GlobalTransform, &mut TerrainChunkView), Changed<Transform>>,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    ) {
        for (entity, g_transform, view) in chunk_views {
            let chunk_half_size = view.chunk_size / 2.;
            // todo: We need a stitch buffer that houses every possible division of stitching required
            // from our QuadTree
            // this makes CHUNK_SIZE vertices at the edge...
            // so 64
            let subdivisions = 32 as u32 - 1;
            // the half res needs to be 32,
            // so 64 / 2 - 1
            let full_res = subdivisions as usize + 2;
            let half_res = (full_res / 2) + 1;
            // todo Terrain Meshes need's to be a component that livex next to our TerrainChunkView
            let terrain_meshes = TerrainMeshes {
                center: meshes.add(
                    Plane3d::new(Vec3::Y, Vec2::splat(chunk_half_size))
                        .mesh()
                        .subdivisions(subdivisions)
                        .build(),
                ),
                north_stitch: meshes.add(Mesh::from(generate_stitched_plane_north(
                    full_res,
                    half_res,
                    full_res,
                    view.chunk_size,
                    view.chunk_size,
                ))),
                east_stitch: meshes.add(Mesh::from(generate_stitched_plane_east(
                    full_res,
                    half_res,
                    full_res,
                    view.chunk_size,
                    view.chunk_size,
                ))),
                south_stitch: meshes.add(Mesh::from(generate_stitched_plane_south(
                    full_res,
                    half_res,
                    full_res,
                    view.chunk_size,
                    view.chunk_size,
                ))),
                west_stitch: meshes.add(Mesh::from(generate_stitched_plane_west(
                    full_res,
                    half_res,
                    full_res,
                    view.chunk_size,
                    view.chunk_size,
                ))),
            };
            commands.entity(entity).insert(terrain_meshes.clone());
            let initially_visible = view.quad.refined_nodes_with_id(
                g_transform
                    .translation()
                    .with_y(g_transform.translation().z)
                    .truncate(),
            );

            for node in view.quad.all() {
                let chunk = node.chunk();
                let quad_scale = chunk.bound.half_size() / chunk_half_size;

                // We create a heights data for each vertex in our mesh I GUESS?
                let heights_data: Vec<f32> = vec![0.; 0];

                let heights_buffer = buffers.add(ShaderStorageBuffer::from(heights_data));

                let (visibility, mesh) = if initially_visible.contains_key(&chunk.id) {
                    let mut neighbours_in_direction = node
                        .neighbours(&initially_visible)
                        .into_iter()
                        .filter(|(_, neighbour_node)| {
                            let neighbour_chunk = neighbour_node.chunk();
                            (chunk.level as i32 - neighbour_chunk.level as i32) < 0
                        })
                        .map(|(neighbouring, _)| neighbouring)
                        .collect::<HashSet<_>>()
                        .into_iter()
                        // todo: Tis v. slow we need t
                        .collect::<Vec<_>>();
                    neighbours_in_direction.sort();
                    let stitch_in = StitchIn::from_neighbour_in(neighbours_in_direction);
                    info!("{:?}", stitch_in);
                    (
                        Visibility::Visible,
                        terrain_meshes.mesh_from_stitch_direction(&stitch_in),
                    )
                } else {
                    (Visibility::Hidden, terrain_meshes.center.clone())
                };

                // todo: We can use batch here
                commands.spawn((
                    Transform::from_translation(
                        chunk
                            .bound
                            .center()
                            .with_y(0.)
                            .extend(chunk.bound.center().y),
                    )
                    .with_scale(Vec3::new(quad_scale.x, 1., quad_scale.y)),
                    TerrainChunk {
                        center: chunk.bound.center(),
                        // todo: chunk.pos is looking somewhat incorrect, we should probably fix
                        // it?
                        pos: chunk.pos(Vec2::splat(view.chunk_size)),
                        size: (chunk.bound.half_size() / (chunk_half_size)).as_ivec2(),
                        heights_buffer: heights_buffer.clone(),
                    },
                    visibility,
                    MeshMaterial3d(terrain_materials.add(TerrainMaterial {
                        height_map: terrain_images.height_map.clone(),
                        ground_textures: terrain_loading.ground.clone(),
                        ground_normals: terrain_loading.ground_normal.clone(),
                        slope_textures: terrain_loading.slope.clone(),
                        slope_normals: terrain_loading.slope_normal.clone(),
                        params: TerrainMaterialParams {
                            offset: (chunk.bound.min / view.terrain_size) * HEIGHT_MAP_SIZE as f32,
                            size: ((chunk.bound.half_size() * 2.) / view.terrain_size)
                                * HEIGHT_MAP_SIZE as f32,
                            height_mult: view.terrain_height,
                        },
                        heights_buffer,
                    })),
                    Wireframe,
                    Mesh3d(mesh),
                ));
                height_map_uniforms.set_changed();
            }
        }
    }

    ///
    /// Updates meshes based on the distance to the camera,
    fn update_meshes(
        mut commands: Commands,
        mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
        terrain_images: Res<TerrainGpuImages>,
        terrain_loading: Res<TerrainLoadingImages>,
        mut height_map_uniforms: ResMut<HeightMapUniforms>,
        chunk_views: Query<
            (&GlobalTransform, &mut TerrainChunkView, &TerrainMeshes),
            Changed<Transform>,
        >,
        chunks: Query<(Entity, &TerrainChunk)>,
        mut images: ResMut<Assets<Image>>,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    ) {
        if !terrain_loading.texture_loaded {
            return;
        }
        for (g_transform, mut view, terrain_meshes) in chunk_views {}
    }
}

#[derive(Component, Clone)]
struct TerrainMeshes {
    center: Handle<Mesh>,
    north_stitch: Handle<Mesh>,
    east_stitch: Handle<Mesh>,
    south_stitch: Handle<Mesh>,
    west_stitch: Handle<Mesh>,
}

impl TerrainMeshes {
    fn mesh_from_stitch_direction(&self, dir: &StitchIn) -> Handle<Mesh> {
        match dir {
            StitchIn::None => self.center.clone(),
            StitchIn::North => self.north_stitch.clone(),
            StitchIn::NorthEast => self.center.clone(),
            StitchIn::East => self.east_stitch.clone(),
            StitchIn::SouthEast => self.center.clone(),
            StitchIn::South => self.south_stitch.clone(),
            StitchIn::SouthWest => self.center.clone(),
            StitchIn::West => self.west_stitch.clone(),
            StitchIn::NorthWest => self.center.clone(),
        }
    }
}

fn startup_terrain(
    asset_server: Res<AssetServer>,
    mut texture_loading: ResMut<TerrainLoadingImages>,
) {
    if texture_loading.texture_loaded || !texture_loading.is_loaded(&asset_server) {
        return;
    }
    // create the image
    texture_loading.texture_loaded = true;
}

pub type BlendMaterial = ExtendedMaterial<StandardMaterial, TextureArrayMaterial>;

#[derive(Clone, ShaderType, Debug)]
struct TerrainMaterialParams {
    offset: Vec2,
    size: Vec2,
    height_mult: f32,
}

#[derive(AsBindGroup, Asset, TypePath, Debug, Clone)]
pub struct TerrainMaterial {
    // tddo: We pass in our desired LoD to the texture here, then we have another uniform for the
    // chunk id to use. We do this so we don't have to bind all the LoD levels to our
    // MaterialShader... but then...
    #[texture(0)]
    #[sampler(1)]
    height_map: Handle<Image>,

    #[texture(3, dimension = "2d_array")]
    #[sampler(4)]
    ground_textures: Handle<Image>,

    #[texture(5, dimension = "2d_array")]
    #[sampler(6)]
    ground_normals: Handle<Image>,

    #[texture(7, dimension = "2d_array")]
    #[sampler(8)]
    slope_textures: Handle<Image>,

    #[texture(9, dimension = "2d_array")]
    #[sampler(10)]
    slope_normals: Handle<Image>,

    #[uniform(11)]
    params: TerrainMaterialParams,

    #[storage(12)]
    heights_buffer: Handle<ShaderStorageBuffer>,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_fragment.wgsl".into()
    }
    // Our displacement shader
    fn vertex_shader() -> ShaderRef {
        "shaders/vertex_height.wgsl".into()
    }
}

/// A custom [`ExtendedMaterial`] that allows blending between the images four channels.
#[derive(AsBindGroup, Asset, TypePath, Default, Debug, Clone)]
pub struct TextureArrayMaterial {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    ground_textures: Handle<Image>,

    #[texture(102, dimension = "2d_array")]
    #[sampler(103)]
    ground_normals: Handle<Image>,

    #[texture(104, dimension = "2d_array")]
    #[sampler(105)]
    slope_textures: Handle<Image>,

    #[texture(106, dimension = "2d_array")]
    #[sampler(107)]
    slope_normals: Handle<Image>,
}

impl TextureArrayMaterial {
    pub fn new(
        ground_textures: Handle<Image>,
        ground_normals: Handle<Image>,
        slope_textures: Handle<Image>,
        slope_normals: Handle<Image>,
    ) -> Self {
        Self {
            ground_textures,
            ground_normals,
            slope_textures,
            slope_normals,
        }
    }
}

impl MaterialExtension for TextureArrayMaterial {
    // we have to use deferred so we can add normals
    fn fragment_shader() -> ShaderRef {
        "shaders/blend_textures.wgsl".into()
    }
}

fn startup_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(TerrainLoadingImages {
        texture_loaded: false,
        ground: asset_server.load_with_settings(
            "textures/ground.ktx2",
            move |s: &mut ImageLoaderSettings| {
                let sampler_desc = ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..Default::default()
                };
                s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
            },
        ),
        // todo: We need to load dx when using the DX pipeline
        ground_normal: asset_server.load_with_settings(
            "textures/ground_normal_gl.ktx2",
            move |s: &mut ImageLoaderSettings| {
                let sampler_desc = ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..Default::default()
                };
                s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
            },
        ),

        slope: asset_server.load_with_settings(
            "textures/slope.ktx2",
            move |s: &mut ImageLoaderSettings| {
                let sampler_desc = ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..Default::default()
                };
                s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
            },
        ),
        // todo: We need to load dx when using the DX pipeline
        slope_normal: asset_server.load_with_settings(
            "textures/slope_normal_gl.ktx2",
            move |s: &mut ImageLoaderSettings| {
                let sampler_desc = ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..Default::default()
                };
                s.sampler = ImageSampler::Descriptor(sampler_desc.clone());
            },
        ),
    });
}
