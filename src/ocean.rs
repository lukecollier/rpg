use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::PipelineCacheError,
};
use std::borrow::Cow;

const SPECTRUM_COMPUTE_SHADER_PATH: &str = "shaders/ocean_compute_2.wgsl";
const HEIGHT: u32 = 64;
const WIDTH: u32 = 64;

pub struct OceanPlugin;

#[derive(Resource, Clone, ExtractResource)]
struct OceanImages {
    debug_output: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct OceanParams {
    time: f32,
    peak_frequency: f32,
    wind_speed: f32,
    grid_size: u32,
}

struct OceanNode {
    state: OceanState,
}

enum OceanState {
    Loading,
    Init,
}

impl Default for OceanNode {
    fn default() -> Self {
        Self {
            state: OceanState::Loading,
        }
    }
}

impl render_graph::Node for OceanNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<OceanPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            OceanState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        info!("loaded success");
                        self.state = OceanState::Init;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets\n{err}")
                    }
                    _ => {}
                }
            }
            OceanState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                }
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        // select the pipeline based on the current state
        match self.state {
            OceanState::Loading => {}
            OceanState::Init => {
                let bind_groups = &world.resource::<OceanImageBindGroups>().0;
                let pipeline_cache = world.resource::<PipelineCache>();
                let pipeline = world.resource::<OceanPipeline>();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_pipeline(init_pipeline);
                let workgroup_size = 8u32;
                let groups_x = (WIDTH + workgroup_size - 1) / workgroup_size;
                let groups_y = (HEIGHT + workgroup_size - 1) / workgroup_size;
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
        }
        Ok(())
    }
}

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup, debug_mesh).chain());

        app.add_plugins((
            ExtractResourcePlugin::<OceanImages>::default(),
            ExtractResourcePlugin::<OceanParams>::default(),
        ));

        app.add_systems(Update, update_ocean_time);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_ocean_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(OceanLabel, OceanNode::default());
        render_graph.add_node_edge(OceanLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Resource)]
struct OceanPipeline {
    texture_bind_group_layout: BindGroupLayout,
    // init_pipeline: CachedComputePipelineId,
    init_pipeline: CachedComputePipelineId,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct OceanLabel;

#[derive(Resource)]
struct OceanImageBindGroups([BindGroup; 1]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<OceanPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    ocean_images: Res<OceanImages>,
    ocean_uniforms: Res<OceanParams>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_debug_output = gpu_images.get(&ocean_images.debug_output).unwrap();

    let mut uniform_buffer = UniformBuffer::from(ocean_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        Some("ocean_bind_group_0"),
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((&view_debug_output.texture_view, &uniform_buffer)),
    );

    commands.insert_resource(OceanImageBindGroups([bind_group_0]));
}

#[derive(Component)]
struct DebugMesh;

fn debug_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    images: Res<OceanImages>,
) {
    commands.spawn((
        DebugMesh,
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            metallic: 0.,
            reflectance: 0.,

            base_color_texture: Some(images.debug_output.clone()),
            ..Default::default()
        })),
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::new(16., 16.)))),
        Transform::from_xyz(0., 25., 0.),
    ));
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let debug_output = images.add(image.clone());

    // create our textures, these will be written by the compute shaders
    commands.insert_resource(OceanImages { debug_output });

    commands.insert_resource(OceanParams {
        time: 0.0,
        peak_frequency: 0.1,
        wind_speed: 2.0,
        grid_size: WIDTH,
    });
}

fn init_ocean_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        "OceanImages",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<OceanParams>(false),
            ),
        ),
    );
    let shader = asset_server.load(SPECTRUM_COMPUTE_SHADER_PATH);
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("init")),
        ..default()
    });

    commands.insert_resource(OceanPipeline {
        texture_bind_group_layout,
        init_pipeline,
    });
}

fn update_ocean_time(
    mut ocean_uniforms: ResMut<OceanParams>,
    time: Res<Time>, // Bevy time resource
) {
    // Update the simulation time
    ocean_uniforms.time += time.delta_secs();
}
