use std::{borrow::Cow, collections::HashSet, time::Duration};

use avian3d::prelude::*;
use bevy::{
    asset::{Asset, Handle, RenderAssetUsages},
    camera::primitives::Aabb,
    color::palettes::css::*,
    gltf::GltfMeshName,
    image::{Image, ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial, wireframe::Wireframe,
    },
    prelude::*,
    reflect::TypePath,
    render::{
        RenderApp, RenderStartup,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            AsBindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType, StorageBuffer,
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
use bytemuck::{Pod, Zeroable};

// todo: Move these into terrain settings, we can have TerrainPlugin init the resource
const CHUNK_SIZE: f32 = 64.;
const TERRAIN_MAX_HEIGHT: f32 = 256.;
const TERRAIN_VIEW_DISTANCE: f32 = 2048. * 2.;
const LOD0_SUBDIVISIONS: u32 = 32;
const LOD0_BUFFER_SIZE: u32 = 128;
const LOD1_BUFFER_SIZE: u32 = 128;
const LOD2_BUFFER_SIZE: u32 = 256;

const HEIGHT_MAP_SIZE: u32 = 128;
const LOD1_HEIGHT_MAP_SIZE: u32 = 64;
const LOD2_HEIGHT_MAP_SIZE: u32 = 16;

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
        render_app.add_systems(RenderStartup, TerrainPipeline::init_pipeline);
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

/// Quad tree implementation that only cares about leaf nodes.
#[derive(Debug, Clone)]
pub struct SinglePointQuadTree {
    nodes: Vec<(Rect, StitchIn)>,
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
    fn from_neighbour_in(neighbour_ins: Vec<NeighbourIn>) -> Self {
        match neighbour_ins.as_slice() {
            [NeighbourIn::Top, NeighbourIn::Left] => Self::NorthWest,
            [NeighbourIn::Top] => Self::North,
            [NeighbourIn::Top, NeighbourIn::Right] => Self::NorthEast,
            [NeighbourIn::Right] => Self::East,
            [NeighbourIn::Right, NeighbourIn::Bottom] => Self::SouthEast,
            [NeighbourIn::Bottom] => Self::South,
            [NeighbourIn::Bottom, NeighbourIn::Left] => Self::SouthWest,
            [NeighbourIn::Left] => Self::West,
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

impl SinglePointQuadTree {
    fn debug(&self, gizmos: &mut Gizmos) {
        for (node, direction) in &self.nodes {
            gizmos.primitive_3d(
                &Cuboid::new(node.half_size().x * 2., 0., node.half_size().y * 2.),
                Isometry3d::from_translation(node.center().with_y(0.).extend(node.center().y)),
                direction.to_color(),
            );
        }
    }

    fn new(bot_left: Vec2, top_right: Vec2, target_size: Vec2, point: Vec2) -> Self {
        let mut tree = Self::init(bot_left, top_right, target_size, point);
        let mut new_tree: Vec<Rect> = Vec::with_capacity(tree.len());
        let mut finished = false;
        while !finished {
            new_tree.clear();

            for (node, neighbours) in tree.clone().iter().zip(Self::neighbours(&tree)) {
                // if our node has met the minimum subdivide we ignore it
                if node.size() == target_size {
                    new_tree.push(*node);
                    continue;
                }
                let mut found = false;
                for (_, neighbour) in neighbours {
                    let resolution_diff =
                        neighbour.size().length_squared() / node.size().length_squared();
                    // resolution difference should be
                    if resolution_diff > 1.0 {
                        found = true;
                        break;
                    }
                }
                if found {
                    let half_size = node.half_size();
                    let center = node.center();
                    let bot_left_quad = Rect::from_corners(node.min, node.min + half_size);
                    let bot_right_quad = Rect::new(
                        center.x,
                        node.min.y,
                        center.x + half_size.x,
                        node.min.y + half_size.y,
                    );
                    let top_right_quad = Rect::from_corners(center, node.max);
                    let top_left_quad = Rect::new(
                        node.min.x,
                        center.y,
                        node.min.x + half_size.x,
                        center.y + half_size.y,
                    );
                    let quads = [bot_left_quad, bot_right_quad, top_left_quad, top_right_quad];
                    new_tree.extend(quads);
                } else {
                    new_tree.push(*node);
                }
            }
            // if our tree hasn't grown then we finish our loop
            if tree.len() == new_tree.len() {
                finished = true;
            }
            tree = new_tree.clone();
        }
        let tree_with_dir =
            tree.clone()
                .into_iter()
                .zip(Self::neighbours(&tree))
                .map(|(a, neighbours)| {
                    let mut valid_neighbours: Vec<NeighbourIn> = neighbours
                        .into_iter()
                        .filter(|(_, b)| {
                            (b.size().length_squared() / a.size().length_squared()) > 1.
                        })
                        .map(|(neighbour_in, _)| neighbour_in)
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect();
                    valid_neighbours.sort();
                    (a, StitchIn::from_neighbour_in(valid_neighbours))
                });
        Self {
            nodes: tree_with_dir.collect(),
        }
    }

    /// todo: Need to add stitchdirection topkek
    fn diff(self, other: SinglePointQuadTree) -> (Vec<(Rect, StitchIn)>, Vec<(Rect, StitchIn)>) {
        let removed = other
            .nodes
            .clone()
            .into_iter()
            .filter(|a| !self.nodes.contains(a))
            .collect();
        let added = self
            .nodes
            .clone()
            .into_iter()
            .filter(|a| !other.nodes.contains(a))
            .collect();
        (added, removed)
    }

    fn init(bot_left: Vec2, top_right: Vec2, target_size: Vec2, point: Vec2) -> Vec<Rect> {
        let origin = Rect::from_corners(bot_left, top_right);
        let center = origin.center();
        let half_size = origin.half_size();
        let threshold = origin.min.distance_squared(center) * 0.50;
        let mut leafs: Vec<Rect> = Vec::with_capacity(4);
        let bot_left_quad = Rect::from_corners(origin.min, origin.min + half_size);
        let bot_right_quad = Rect::new(
            center.x,
            origin.min.y,
            center.x + half_size.x,
            origin.min.y + half_size.y,
        );
        let top_right_quad = Rect::from_corners(center, origin.max);
        let top_left_quad = Rect::new(
            origin.min.x,
            center.y,
            origin.min.x + half_size.x,
            center.y + half_size.y,
        );
        let quads = [bot_left_quad, bot_right_quad, top_left_quad, top_right_quad];

        // todo: We want to do this in breadth first order,
        // this is so when we diff later on we can quickly determine where need's updating.
        // so if our top left quad hasn't updated we will zip together all our quads and validate
        // which specific quads have been updated more quickly.
        for quad in quads {
            if quad.center().distance_squared(point) < threshold
                && quad.size().x > target_size.x
                && quad.size().y > target_size.y
            {
                let children = Self::init(quad.min, quad.max, target_size, point);
                leafs.extend(children);
            } else {
                leafs.push(quad);
            }
        }
        leafs
    }

    /// Returns the direction *from r1 to r2* if they border each other.
    /// Returns None if they do not touch or only touch at a corner.
    pub fn bordering_direction(r1: &Rect, r2: &Rect) -> Option<NeighbourIn> {
        // Convenience
        let r1_left = r1.min.x;
        let r1_right = r1.max.x;
        let r1_bottom = r1.min.y;
        let r1_top = r1.max.y;

        let r2_left = r2.min.x;
        let r2_right = r2.max.x;
        let r2_bottom = r2.min.y;
        let r2_top = r2.max.y;

        // --- Shared edge overlap helpers ---
        let overlap_x = r1_left < r2_right && r1_right > r2_left;
        let overlap_y = r1_bottom < r2_top && r1_top > r2_bottom;

        // --- Check top adjacency (r2 is directly above r1) ---
        if (r1_top == r2_bottom) && overlap_x {
            return Some(NeighbourIn::Top);
        }

        // --- Check bottom adjacency (r2 is directly below r1) ---
        if (r1_bottom == r2_top) && overlap_x {
            return Some(NeighbourIn::Bottom);
        }

        // --- Check right adjacency (r2 is directly right of r1) ---
        if (r1_right == r2_left) && overlap_y {
            return Some(NeighbourIn::Right);
        }

        // --- Check left adjacency (r2 is directly left of r1) ---
        if (r1_left == r2_right) && overlap_y {
            return Some(NeighbourIn::Left);
        }

        None
    }

    /// Using this method we can determine when to drop resolutions
    fn neighbours(tree: &Vec<Rect>) -> Vec<Vec<(NeighbourIn, Rect)>> {
        tree.iter()
            .map(|rect| {
                tree.iter()
                    .filter_map(|other_rect| {
                        Self::bordering_direction(rect, other_rect)
                            .map(|direction| (direction, *other_rect))
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod quad_tree_tests {
    use bevy::math::{Rect, Vec2};

    use super::NeighbourIn;
    use super::SinglePointQuadTree;

    #[test]
    fn new_fix() {
        let r = Rect {
            min: Vec2::new(-1024.0, -2048.0),
            max: Vec2::new(0.0, -1024.0),
        };
        let r1 = Rect {
            min: Vec2::new(-2048.0, -2048.0),
            max: Vec2::new(-1024.0, -1024.0),
        };
        let r2 = Rect {
            min: Vec2::new(-1024.0, -1024.0),
            max: Vec2::new(-512.0, -512.0),
        };
        let r3 = Rect {
            min: Vec2::new(-512.0, -1024.0),
            max: Vec2::new(0.0, -512.0),
        };
        let r4 = Rect {
            min: Vec2::new(0.0, -2048.0),
            max: Vec2::new(1024.0, -1024.0),
        };
        assert_eq!(
            SinglePointQuadTree::bordering_direction(&r, &r1),
            Some(NeighbourIn::Left)
        );
        assert_eq!(
            SinglePointQuadTree::bordering_direction(&r, &r2),
            Some(NeighbourIn::Top)
        );
        assert_eq!(SinglePointQuadTree::bordering_direction(&r, &r3), None);
        assert_eq!(
            SinglePointQuadTree::bordering_direction(&r, &r4),
            Some(NeighbourIn::Right)
        );
    }

    #[test]
    fn ensure_fix() {
        let r1 = Rect::new(0., 0., 32., 32.);
        let r2 = Rect::new(64., 0., 96., 32.);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r1, &r2), None);
    }

    #[test]
    fn no_corners() {
        let r1 = Rect::new(32., 32., 64., 64.);
        let r2 = Rect::new(0., 0., 32., 32.);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r1, &r2), None);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r2, &r1), None);
        let r3 = Rect::new(-32., 32., 0., 64.);
        let r4 = Rect::new(0., 0., 32., 32.);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r3, &r4), None);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r4, &r3), None);
    }

    #[test]
    fn overlapping() {
        let r1 = Rect::new(96., 96., 128., 128.);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r1, &r1), None);
    }

    #[test]
    fn bordering_none() {
        let r1 = Rect::new(96., 96., 128., 128.);
        let r2 = Rect::new(0., 0., 64., 64.);
        assert_eq!(SinglePointQuadTree::bordering_direction(&r1, &r2), None);
    }

    #[test]
    fn bordering_bot_wider() {
        let y = 64.;
        let r1 = Rect::new(0., y, 64., 128.);
        let r2 = Rect::new(0., 0., 64., y);
        assert_eq!(
            SinglePointQuadTree::bordering_direction(&r1, &r2),
            Some(NeighbourIn::Bottom)
        );
    }

    #[test]
    fn bordering_top_wider() {
        let y = 64.;
        let r1 = Rect::new(0., 0., 64., y);
        let r2 = Rect::new(0., y, 64., y + 64.);
        assert_eq!(
            SinglePointQuadTree::bordering_direction(&r1, &r2),
            Some(NeighbourIn::Top)
        );
    }
}

enum TerrainRenderState {
    /// Ensure init scripts are loaded
    Loading,
    /// Updates a height map at (pos, buffer_index)
    Update(Vec<TerrainChunk>),
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
                        // todo: We want to check if the chunks have changed and cause a redraw, but
                        // technically terrain buffer indexes handles this (for now)
                        let chunks = world
                            .query::<&TerrainChunk>()
                            .iter(&world)
                            .cloned()
                            .collect::<Vec<_>>();
                        self.state = TerrainRenderState::Update(chunks);
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
            TerrainRenderState::Update(_) => {
                if !world.is_resource_changed::<HeightMapUniforms>() {
                    self.state = TerrainRenderState::Monitor;
                }
            }
            TerrainRenderState::Monitor => {
                if world.is_resource_changed::<HeightMapUniforms>() {
                    let chunks = world
                        .query::<&TerrainChunk>()
                        .iter(&world)
                        .cloned()
                        .collect::<Vec<_>>();
                    self.state = TerrainRenderState::Update(chunks);
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
        let pipeline_cache = world.resource::<PipelineCache>();
        let terrain_pipeline = world.resource::<TerrainPipeline>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let queue = world.resource::<RenderQueue>();

        match &self.state {
            TerrainRenderState::Loading => {}
            TerrainRenderState::Coarse => {}
            TerrainRenderState::Update(chunks) => {
                let pipeline = pipeline_cache
                    .get_compute_pipeline(terrain_pipeline.height_map_pipeline)
                    .unwrap();
                for chunk in chunks {
                    let height_map_gpu = gpu_images.get(&chunk.gpu_image).unwrap();
                    let size = height_map_gpu.size_2d();
                    let mut uniform_buffer = UniformBuffer::from(HeightMapUniforms {
                        frequency: height_map_params.frequency,
                        lacunarity: height_map_params.lacunarity,
                        octaves: height_map_params.octaves,
                        persistence: height_map_params.persistence,
                        x_from: (chunk.pos.x as f32) * 2.,
                        x_to: (chunk.pos.x as f32 + chunk.size.x as f32) * 2.,
                        y_from: (chunk.pos.y as f32) * 2.,
                        y_to: (chunk.pos.y as f32 + chunk.size.y as f32) * 2.,
                    });
                    uniform_buffer.write_buffer(&render_context.render_device(), &queue);

                    let bind_group = render_context.render_device().create_bind_group(
                        Some("height_map_bind_group"),
                        &terrain_pipeline.texture_bind_group_layout,
                        &BindGroupEntries::sequential((
                            &height_map_gpu.texture_view,
                            &uniform_buffer,
                        )),
                    );

                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(size.x / 8, size.y / 8, 1);
                }
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
    texture_bind_group_layout: BindGroupLayout,
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
            texture_bind_group_layout: bind_group_layout,
            height_map_pipeline: pipeline,
        });
    }
}

// these textures get uploaded to the GPU
#[derive(Resource, Clone, ExtractResource)]
struct TerrainGpuImages {
    /// This is our global height maps for our quad trees,
    /// when a new quad tree is created we generate the height map once and store it.
    /// I GUESS!?
    height_maps: Handle<Image>,
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
                let lod0_handle = height_maps.lod0_height_map.clone();
                commands
                    .spawn(Readback::texture(terrain_images.height_maps.clone()))
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
        mut meshes: ResMut<Assets<Mesh>>,
        asset_server: Res<AssetServer>,
    ) {
        let mut height_map =
            Image::new_target_texture(HEIGHT_MAP_SIZE, HEIGHT_MAP_SIZE, TextureFormat::Rgba32Float);
        height_map.asset_usage = RenderAssetUsages::RENDER_WORLD;
        height_map.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC;
        // height_map.sampler = ImageSampler::nearest();

        commands.insert_resource(TerrainGpuImages {
            height_maps: images.add(height_map.clone()),
            cpu_update_timer: Some(Timer::new(
                Duration::new(HEIGHTMAP_DEBOUNCE_SECS, 0),
                TimerMode::Once,
            )),
        });
        // todo: We need a stitch buffer that houses every possible division of stitching required
        // from our QuadTree
        // this makes CHUNK_SIZE vertices at the edge...
        // so 64
        let subdivisions = 8 as u32 - 1;
        // the half res needs to be 32,
        // so 64 / 2 - 1
        let full_res = subdivisions as usize + 2;
        let half_res = (full_res / 2) + 1;
        commands.insert_resource(TerrainMeshes {
            center: meshes.add(
                Plane3d::new(Vec3::Y, Vec2::splat(CHUNK_SIZE))
                    .mesh()
                    .subdivisions(subdivisions)
                    .build(),
            ),
            north_stitch: meshes.add(Mesh::from(generate_stitched_plane_north(
                full_res,
                half_res,
                full_res,
                CHUNK_SIZE * 2.,
                CHUNK_SIZE * 2.,
            ))),
            east_stitch: meshes.add(Mesh::from(generate_stitched_plane_east(
                full_res,
                half_res,
                full_res,
                CHUNK_SIZE * 2.,
                CHUNK_SIZE * 2.,
            ))),
            south_stitch: meshes.add(Mesh::from(generate_stitched_plane_south(
                full_res,
                half_res,
                full_res,
                CHUNK_SIZE * 2.,
                CHUNK_SIZE * 2.,
            ))),
            west_stitch: meshes.add(Mesh::from(generate_stitched_plane_west(
                full_res,
                half_res,
                full_res,
                CHUNK_SIZE * 2.,
                CHUNK_SIZE * 2.,
            ))),
        });
        // we need to calculate the overlap
        commands.insert_resource(HeightMapUniforms {
            frequency: terrain_settings.frequency,
            lacunarity: terrain_settings.lacunarity,
            octaves: terrain_settings.octaves as i32,
            persistence: terrain_settings.persistence,
            x_from: terrain_settings.x_bounds.0 as f32,
            x_to: terrain_settings.x_bounds.1 as f32,
            y_from: terrain_settings.y_bounds.0 as f32,
            y_to: terrain_settings.y_bounds.1 as f32,
        });
    }
}

// these heightmaps exist only on the CPU and are read back from the GPU on change.
// these height maps only use the R channel compared to the GPU which stores the derivatives as
// well.
#[derive(Resource, Clone)]
struct TerrainHeightMaps {
    lod0_height_map: Handle<Image>,
    lod1_height_map: Handle<Image>,
    lod2_height_map: Handle<Image>,
}

impl TerrainHeightMaps {
    fn startup(mut commands: Commands, mut assets: ResMut<Assets<Image>>) {
        let mut lod0_image = Image::new_target_texture(
            HEIGHT_MAP_SIZE,
            HEIGHT_MAP_SIZE * LOD0_BUFFER_SIZE,
            TextureFormat::Rgba32Float,
        );
        lod0_image.asset_usage = RenderAssetUsages::MAIN_WORLD;
        lod0_image.reinterpret_stacked_2d_as_array(LOD0_BUFFER_SIZE);

        let mut lod1_image = Image::new_target_texture(
            LOD1_HEIGHT_MAP_SIZE,
            LOD1_HEIGHT_MAP_SIZE * LOD1_BUFFER_SIZE,
            TextureFormat::Rgba32Float,
        );
        lod1_image.asset_usage = RenderAssetUsages::MAIN_WORLD;
        lod1_image.reinterpret_stacked_2d_as_array(LOD1_BUFFER_SIZE);

        let mut lod2_image = Image::new_target_texture(
            LOD2_HEIGHT_MAP_SIZE,
            LOD2_HEIGHT_MAP_SIZE * LOD2_BUFFER_SIZE,
            TextureFormat::Rgba32Float,
        );
        lod2_image.asset_usage = RenderAssetUsages::MAIN_WORLD;
        lod2_image.reinterpret_stacked_2d_as_array(LOD2_BUFFER_SIZE);
        commands.insert_resource(TerrainHeightMaps {
            lod0_height_map: assets.add(lod0_image),
            lod1_height_map: assets.add(lod1_image),
            lod2_height_map: assets.add(lod2_image),
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
        let half = CHUNK_SIZE;
        let full = CHUNK_SIZE * 2.;
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
    pub direction: StitchIn,
    pub pos: IVec2,
    pub size: IVec2,
    // heightmap to populate
    pub gpu_image: Handle<Image>,
    // heightmap readback
    pub image: Option<Handle<Image>>,
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
    quad: SinglePointQuadTree,
    view_distance: f32,
}

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum NeighbourIn {
    Top,
    Right,
    Bottom,
    Left,
}

impl TerrainChunkView {
    fn update_debug_quad_tree(
        chunk_views: Query<&TerrainChunkView, Changed<Transform>>,
        mut gizmos: Gizmos,
    ) {
        for view in chunk_views {
            view.quad.debug(&mut gizmos);
        }
    }

    pub fn new() -> Self {
        Self {
            quad: SinglePointQuadTree { nodes: vec![] },
            view_distance: TERRAIN_VIEW_DISTANCE,
        }
    }

    // todo: To make this more fluid we actually need to sample the point the player is moving
    // towards as well by approx the smallest chunk's size. this way we can load in the high
    // resolution across boarders preventing pop-in
    fn new_quad_tree(&mut self, position: Vec2) -> SinglePointQuadTree {
        let bot_left = Vec2::splat(-self.view_distance / 2.);
        let top_right = Vec2::splat(self.view_distance / 2.);
        // todo: If we make this 3d we can allow flying lel
        // now we can use a single instanced mesh for all of our geometry.
        // The size of the quad tree will dictate the geometry
        SinglePointQuadTree::new(bot_left, top_right, Vec2::splat(CHUNK_SIZE * 2.), position)
    }

    ///
    /// Updates meshes based on the distance to the camera,
    fn update_meshes(
        mut commands: Commands,
        mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
        terrain_meshes: Res<TerrainMeshes>,
        terrain_images: Res<TerrainGpuImages>,
        terrain_loading: Res<TerrainLoadingImages>,
        mut height_map_uniforms: ResMut<HeightMapUniforms>,
        chunk_views: Query<(&GlobalTransform, &mut TerrainChunkView), Changed<Transform>>,
        chunks: Query<(Entity, &TerrainChunk)>,
        mut images: ResMut<Assets<Image>>,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    ) {
        if !terrain_loading.texture_loaded {
            return;
        }
        for (g_transform, mut view) in chunk_views {
            let quad_tree = view.new_quad_tree(
                // Vec2::ZERO,
                g_transform
                    .translation()
                    .with_y(g_transform.translation().z)
                    .truncate(),
            );
            let (added, removed) = quad_tree.clone().diff(view.quad.clone());
            view.quad = quad_tree;

            for (entity, chunk) in chunks {
                if removed
                    .iter()
                    .find(|(rect, direction)| {
                        rect.center() == chunk.center && chunk.direction == *direction
                    })
                    .is_some()
                {
                    commands.entity(entity).despawn();
                }
            }
            // let origin = (g_transform.translation().truncate() / CHUNK_SIZE).as_ivec2();
            for (quad, direction) in added {
                height_map_uniforms.set_changed();
                let quad_scale = quad.size() / (CHUNK_SIZE * 2.);
                // // todo: We can probably get away with a decently high resolution texture across
                // // eac quad
                let mut gpu_image = Image::new_target_texture(
                    (CHUNK_SIZE * 2.) as u32 as u32,
                    (CHUNK_SIZE * 2.) as u32 as u32,
                    TextureFormat::Rgba32Float,
                );
                gpu_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
                gpu_image.texture_descriptor.dimension =
                    bevy::render::render_resource::TextureDimension::D2;
                gpu_image.texture_descriptor.usage = TextureUsages::COPY_DST
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC;
                // gpu_image.sampler = ImageSampler::nearest();
                let image_handle = images.add(gpu_image);

                // We create a heights data for each vertex in our mesh I GUESS?
                let heights_data: Vec<f32> = vec![0.; 4225 + 1];

                let heights_buffer = buffers.add(ShaderStorageBuffer::from(heights_data));

                let mesh = terrain_meshes.mesh_from_stitch_direction(&direction);

                // todo: We can use batch here
                commands.spawn((
                    // todo: We should only need to spawn the collider based on the detail we
                    // want to load
                    // TerrainCollider {
                    //     // the terrain_height_maps lives on the cpu and is updated
                    //     // automatically with Readbacks from the GPU, snazzy
                    //     height_map: terrain_height_maps.lod0_height_map.clone(),
                    //     index: next_index,
                    //     mesh: terrain_meshes.high_detail.clone(),
                    // },
                    Transform::from_translation(quad.center().with_y(0.).extend(quad.center().y))
                        .with_scale(Vec3::new(quad_scale.x, 1., quad_scale.y)),
                    TerrainChunk {
                        direction: direction,
                        center: quad.center(),
                        pos: (quad.min / (CHUNK_SIZE * 2.)).as_ivec2(),
                        size: (quad.size() / (CHUNK_SIZE * 2.)).as_ivec2(),
                        gpu_image: image_handle.clone(),
                        // will be readback to from the GPU
                        image: None,
                        heights_buffer: heights_buffer.clone(),
                    },
                    Visibility::Inherited,
                    MeshMaterial3d(terrain_materials.add(TerrainMaterial {
                        height_map: image_handle.clone(),
                        ground_textures: terrain_loading.ground.clone(),
                        ground_normals: terrain_loading.ground_normal.clone(),
                        slope_textures: terrain_loading.slope.clone(),
                        slope_normals: terrain_loading.slope_normal.clone(),
                        height_mult: TERRAIN_MAX_HEIGHT,
                        heights_buffer,
                    })),
                    // Wireframe,
                    Mesh3d(mesh),
                ));
            }
        }
    }
}

#[derive(Resource)]
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

// EZ, we actually get the position from the entity itself, here just handles which buffer indexes
// are currently in use by our entities. this is neato! Since we can just request a new

// struct TerrainBufferIndex {
//     max: u32,
//     // a list of the chunks we want to spawn and their LoD
//     // requested: Vec<(IVec2, LOD)>,
//     in_use: HashSet<u32>,
// }

// impl TerrainBufferIndex {
//     fn next_free_index(&self) -> Option<u32> {
//         for idx in 0..self.max {
//             if self.in_use.get(&idx).is_none() {
//                 return Some(idx);
//             }
//         }
//         return None;
//     }

//     /// Claims a buffer position from the available positions.
//     fn claim(&mut self) -> Option<u32> {
//         let index = self.next_free_index()?;
//         self.in_use.insert(index);
//         Some(index)
//     }

//     fn free(&mut self, index: u32) {
//         self.in_use.remove(&index);
//     }
// }

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
    height_mult: f32,

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
