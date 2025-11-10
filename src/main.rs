use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::TEAL,
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
    feathers::{
        FeathersPlugins,
        controls::{SliderProps, slider},
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    light::NotShadowCaster,
    log::LogPlugin,
    mesh::VertexAttributeValues,
    pbr::{ExtendedMaterial, MaterialExtension, wireframe::WireframePlugin},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat},
        view::Hdr,
    },
    shader::ShaderRef,
    ui_widgets::{SliderPrecision, SliderStep, ValueChange, observe, slider_self_update},
};
use bevy_flycam::prelude::*;
use bevy_world_gen::terrain::{BlendMaterial, TerrainPlugin, create_material};
use noise::*;

const WATER_SHADER_ASSET_PATH: &str = "shaders/water_material.wgsl";
const HEIGHT: f64 = 128.;
const WIDTH: f64 = 128.;
const MAGNITUDE: f32 = 50.;
const CHUNK_SIZE: f32 = 128.;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            // filter: "wgpu=trace,warn".into(),
            // level: bevy::log::Level::TRACE,
            ..Default::default()
        }))
        .add_plugins(TerrainPlugin)
        .add_plugins(WireframePlugin::default())
        .add_plugins(FeathersPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        // .add_plugins(OceanPlugin)
        .add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, Water>>::default())
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(Controls::default())
        .insert_resource(TerrainGenerator::default())
        .insert_resource(MeshRef::default())
        .insert_resource(ShoreTexture::default())
        .insert_resource(DebugShoreTextureRef::default())
        .init_resource::<TerrainMaterials>()
        // terrain systems
        .add_systems(Update, (startup_terrain, update_terrain).chain())
        .add_systems(
            Startup,
            (
                startup_textures,
                startup_light,
                startup_ui,
                // startup_water,
                startup_alpha_texture,
                startup_camera,
            ),
        )
        .add_systems(
            Update,
            (
                update_shore_texture,
                // update_shader_water,
                TerrainGenerator::update_generator,
            ),
        )
        .run();
}

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
#[derive(AsBindGroup, Asset, TypePath, Default, Debug, Clone)]
struct Water {
    #[uniform(100)]
    pub params: WaterParams,
    // #[texture(101)]
    // pub depth_texture: Handle<Image>,
}

impl MaterialExtension for Water {
    // fn fragment_shader() -> ShaderRef {
    //     SHADER_ASSET_PATH.into()
    // }

    // fn deferred_fragment_shader() -> ShaderRef {
    //     DEFERRED_WATER_SHADER.into()
    // }

    fn vertex_shader() -> ShaderRef {
        WATER_SHADER_ASSET_PATH.into()
    }
}

/// Parameters to the water shader.
#[derive(ShaderType, Copy, Default, Debug, Clone)]
struct WaterParams {
    amplitude: f32,
    frequency: f32,
    speed: f32,
    phase: f32,
    time: f32,
}

fn startup_water(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut water_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, Water>>>,
) {
    commands.spawn((
        Mesh3d(
            meshes.add(
                Plane3d::new(Vec3::Y, Vec2::splat(CHUNK_SIZE / 2.))
                    .mesh()
                    .subdivisions(1024),
            ),
        ),
        NotShadowCaster,
        MeshMaterial3d(water_materials.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: TEAL.into(),
                perceptual_roughness: 0.0,
                ior: 1.33,
                alpha_mode: AlphaMode::Blend,
                metallic: 0.0,
                ..default()
            },
            extension: Water {
                params: WaterParams {
                    amplitude: 0.03,
                    frequency: 15.0,
                    speed: 2.0,
                    phase: 1. * 17.5,
                    ..default()
                },
            },
        })),
    ));
}

/// Update the fish wobble parameters every frame
/// to animate the wobble effect in the shader.
fn update_shader_water(
    mut mats: ResMut<Assets<ExtendedMaterial<StandardMaterial, Water>>>,
    time: Res<Time>,
) {
    for (_id, mat) in mats.iter_mut() {
        mat.extension.params.time = time.elapsed_secs();
    }
}

fn app_slider(
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    precision: i32,
    op: fn(ResMut<Controls>, f32) -> (),
) -> impl Bundle {
    (
        Node {
            width: percent(25),
            ..default()
        },
        children![
            (Text::new(label), ThemedText),
            (
                slider(
                    SliderProps { min, max, value },
                    (SliderStep(1.), SliderPrecision(precision)),
                ),
                observe(slider_self_update),
                observe(make_slider(op)) // Spawn((Text::new("Normal"), ThemedText)),
            )
        ],
    )
}

fn make_slider(
    op: fn(ResMut<Controls>, f32) -> (),
) -> impl FnMut(On<ValueChange<f32>>, ResMut<Controls>) -> () {
    move |value_change, controls| {
        op(controls, value_change.value);
    }
}

#[derive(Resource, Default)]
struct MeshRef(Option<Handle<Mesh>>);

#[derive(Resource, Default)]
struct LoadingGroundTextures {
    loading_texture: bool,
    ground: Handle<Image>,
    ground_normal: Handle<Image>,

    slope: Handle<Image>,
    slope_normal: Handle<Image>,
}

impl LoadingGroundTextures {
    fn is_loaded(&self, asset_server: &AssetServer) -> bool {
        asset_server.is_loaded(&self.ground)
            && asset_server.is_loaded(&self.ground_normal)
            && asset_server.is_loaded(&self.slope)
            && asset_server.is_loaded(&self.slope_normal)
    }
}

// texture for computing how transparent water should be.
#[derive(Resource, Default)]
struct ShoreTexture(Option<Vec<u8>>);

// texture for computing how transparent water should be.
#[derive(Resource, Default)]
struct DebugShoreTextureRef(Option<Handle<StandardMaterial>>);

// texture for computing how transparent water should be.
#[derive(Resource, Clone, Default)]
struct TerrainMaterials {
    material: Handle<BlendMaterial>,
}

#[derive(Resource, Debug)]
struct TerrainGenerator {
    // todo: We can use a LoD modifier here, basically the further we are away the less octaves we
    // will use. Similarly the mesh can use a lot less geometry when it's far away from the camera,
    // we can do this via chunking as tesselation is not added to bevy yet.
    pub noise: Fbm<OpenSimplex>,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
    pub mid_point: Vec2,
    pub max_dist: f32,
    pub width: f32,
    pub height: f32,
}

impl TerrainGenerator {
    fn update_generator(mut generator: ResMut<TerrainGenerator>, controls: Res<Controls>) {
        if controls.is_changed() {
            let noise = Fbm::<OpenSimplex>::new(0)
                .set_seed(0)
                .set_frequency(controls.frequency.into())
                .set_octaves(controls.octaves as usize)
                .set_persistence(controls.persistence.into())
                .set_lacunarity(controls.lacunarity.into());
            generator.noise = noise;
            generator.x_bounds = controls.x_bounds;
            generator.x_bounds = controls.x_bounds;
        }
    }

    fn get(&self, x: f32, y: f32) -> f32 {
        let x_bounds = self.x_bounds;
        let y_bounds = self.y_bounds;

        let x_extent = x_bounds.1 - x_bounds.0;
        let y_extent = y_bounds.1 - y_bounds.0;

        let x_step = x_extent / self.width as f64;
        let y_step = y_extent / self.height as f64;

        let mid_point = self.mid_point;
        let max_dist = self.max_dist;
        let current_y = y_bounds.0 + y_step * y as f64;
        let current_x = x_bounds.0 + x_step * x as f64;

        let point = Vec2::new(x, y);
        let distance_to_center = (point.distance_squared(mid_point) / max_dist).clamp(0., 1.);

        // todo: Add back the falloff generation
        let val = self.noise.get([current_x, current_y]); // - distance_to_center as f64 * 0.5;

        val as f32
    }
}

impl Default for TerrainGenerator {
    fn default() -> Self {
        let controls = Controls::default();
        let noise = Fbm::<OpenSimplex>::new(0)
            .set_seed(0)
            .set_frequency(controls.frequency.into())
            .set_octaves(controls.octaves as usize)
            .set_persistence(controls.persistence.into())
            .set_lacunarity(controls.lacunarity.into());
        let width = WIDTH as f32;
        Self {
            noise,
            x_bounds: (-5., 5.),
            y_bounds: (-5., 5.),
            mid_point: Vec2::ZERO,
            max_dist: Vec2::ZERO.distance_squared(Vec2::new(width / 2., 0.)),
            width,
            height: HEIGHT as f32,
        }
    }
}

#[derive(Resource, Debug)]
struct Controls {
    pub frequency: f32,
    pub lacunarity: f32,
    pub octaves: f32,
    pub persistence: f32,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
}

impl Default for Controls {
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

fn startup_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            padding: UiRect::px(6., 6., 6., 6.),
            width: percent(100),
            height: px(32),
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            row_gap: px(10),
            ..default()
        },
        ThemeBackgroundColor(tokens::WINDOW_BG),
        children![
            app_slider("frequency", 0.3, 0.1, 1., 1, |ref mut controls, value| {
                controls.frequency = value
            }),
            app_slider("lacunarity", 1.4, 0.1, 3., 1, |ref mut controls, value| {
                controls.lacunarity = value
            }),
            app_slider("octaves", 4., 0., 11., 0, |ref mut controls, value| {
                controls.octaves = value
            }),
            app_slider("persistence", 0.6, 0.0, 1., 1, |ref mut controls, value| {
                controls.persistence = value
            }),
            app_slider("upper", 5., -25., 25., 0, |ref mut controls, value| {
                controls.x_bounds.1 = value as f64;
                controls.y_bounds.1 = value as f64;
            }),
            app_slider("lower", -5., -25., 25., 0, |ref mut controls, value| {
                controls.x_bounds.0 = value as f64;
                controls.y_bounds.0 = value as f64;
            }),
        ],
    ));
}

fn update_terrain(
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_ref: Res<MeshRef>,
    terrain_materials: Res<TerrainMaterials>,
    generator: Res<TerrainGenerator>,
    mut materials: ResMut<Assets<BlendMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(ref mesh_ref) = mesh_ref.0 else {
        return;
    };
    if !generator.is_changed() {
        return;
    }

    if let Some(material) = materials.get_mut(&terrain_materials.material) {
        let width = WIDTH as u32;
        let height = HEIGHT as u32;
        let mut pixel_data: Vec<f32> = vec![0.; (width * height) as usize * 4];
        let mut idx = 1;
        for amount in (0..pixel_data.len()).step_by(4) {
            let x = (idx as f32 % width as f32) - (width as f32 / 2.);
            let y = ((idx as f32 / height as f32).floor() as f32) - (height as f32 / 2.);
            let height = generator.get(x, y);
            idx += 1;
            if let Some(slice) = pixel_data.get_mut(amount..(amount + 4)) {
                if height > 0.20 {
                    slice[3] = 1.;
                } else if height > 0.10 {
                    slice[2] = 1.;
                } else if height > 0. {
                    slice[1] = 1.;
                } else {
                    slice[0] = 1.;
                }
            } else {
                panic!("bad dog");
            }
        }
        let blend_mask: Image = Image::new(
            bevy::render::render_resource::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&pixel_data).to_vec(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        material.base.base_color_texture = Some(images.add(blend_mask))
    }

    let plane = meshes.get_mut(mesh_ref).expect("failed to find mesh");
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        plane.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for pos in positions.iter_mut() {
            let x = pos[0];
            let y = pos[2];

            let val = generator.get(x, y);
            pos[1] = (val) as f32 * MAGNITUDE;
        }
    }
    plane.compute_normals();
}

fn startup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BlendMaterial>>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut mesh_ref: ResMut<MeshRef>,
    mut texture_loading: ResMut<LoadingGroundTextures>,
    mut mesh_generator: ResMut<TerrainGenerator>,
    mut terrain_materials: ResMut<TerrainMaterials>,
) {
    if texture_loading.loading_texture || !texture_loading.is_loaded(&asset_server) {
        return;
    }
    let width = WIDTH as u32;
    let height = HEIGHT as u32;
    let mut pixel_data: Vec<f32> = vec![0.; (width * height) as usize * 4];
    let mut idx = 1;
    for amount in (0..pixel_data.len()).step_by(4) {
        let x = (idx as f32 % width as f32) - (width as f32 / 2.);
        let y = ((idx as f32 / height as f32).floor() as f32) - (height as f32 / 2.);
        let height = mesh_generator.get(x, y);
        // first we normalize between 0-1 I GUESS.
        idx += 1;
        if let Some(slice) = pixel_data.get_mut(amount..(amount + 4)) {
            if height > 0.80 {
                slice[0] = height;
            } else if height > 0.5 {
                slice[1] = height;
            } else if height > 0.15 {
                slice[2] = height;
            } else {
                slice[3] = height;
            }
        } else {
            panic!("bad dog");
        }
    }
    // create the image
    let blend_mask: Image = Image::new(
        bevy::render::render_resource::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(&pixel_data).to_vec(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    texture_loading.loading_texture = true;
    mesh_generator.set_changed();
    let array_layers = 4;
    let texture = images.get_mut(&texture_loading.ground).unwrap();
    texture.reinterpret_stacked_2d_as_array(array_layers);

    let normal = images.get_mut(&texture_loading.ground_normal).unwrap();
    normal.reinterpret_stacked_2d_as_array(array_layers);

    let texture = images.get_mut(&texture_loading.slope).unwrap();
    texture.reinterpret_stacked_2d_as_array(array_layers);

    let normal = images.get_mut(&texture_loading.slope_normal).unwrap();
    normal.reinterpret_stacked_2d_as_array(array_layers);

    let plane = Mesh::from(
        Plane3d::default()
            .mesh()
            .size(CHUNK_SIZE, CHUNK_SIZE)
            .subdivisions(256),
    );
    let asset_id = meshes.add(plane);
    mesh_ref.0 = Some(asset_id.clone());
    let material = create_material(
        images.add(blend_mask).clone(),
        texture_loading.ground.clone(),
        texture_loading.ground_normal.clone(),
        texture_loading.slope.clone(),
        texture_loading.slope_normal.clone(),
    );
    let material = materials.add(material);
    terrain_materials.material = material.clone();
    // plane
    commands.spawn((
        Mesh3d(asset_id),
        // Wireframe,
        MeshMaterial3d(material),
    ));
}

fn startup_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(LoadingGroundTextures {
        loading_texture: false,
        ground: asset_server.load_with_settings(
            "textures/ground.png",
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
            "textures/ground_normal_gl.png",
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
            "textures/slope.png",
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
            "textures/slope_normal_gl.png",
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

fn update_shore_texture(
    mut images: ResMut<Assets<Image>>,
    shore_texture: ResMut<ShoreTexture>,
    material_ref: ResMut<DebugShoreTextureRef>,
    generator: Res<TerrainGenerator>,
) {
    if !generator.is_changed() {
        return;
    }
    if let Some(pixel_data) = &shore_texture.0 {}
}

fn startup_alpha_texture(
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shore_texture: ResMut<ShoreTexture>,
    mut material_ref: ResMut<DebugShoreTextureRef>,
    generator: Res<TerrainGenerator>,
) {
    let width = WIDTH as u32;
    let height = HEIGHT as u32;
    let mut pixel_data = vec![255; (width * height) as usize * 4];
    let mut idx = 1;
    for amount in (0..pixel_data.len()).step_by(4) {
        let x = (idx as f32 % width as f32) - (width as f32 / 2.);
        let y = ((idx as f32 / height as f32).floor() as f32) - (height as f32 / 2.);
        let height = generator.get(x, y);
        idx += 1;
        if let Some(slice) = pixel_data.get_mut(amount..(amount + 4)) {
            let min = -0.20;
            let max = 0.02;
            if height <= max && height > min {
                let normalized = 1.0 - ((height - min) / (max - min));
                // we can eventually use these other channels for other things, like wave intensity
                // etcs.
                slice[0] = 0;
                slice[1] = 0;
                // the blue can denote how deep the water is.
                slice[2] = (255. * normalized) as u8;
                // we only really use the alpha channel, the blue bit is just for debug (for now)
                slice[3] = (255. * normalized) as u8;
            } else if height < min {
                slice[0] = 0;
                slice[1] = 0;
                slice[2] = 255;
                // we only really use the alpha channel, the blue bit is just for debug (for now)
                slice[3] = 255;
            } else {
                slice[3] = 0;
            }
        } else {
            warn!("bad dog");
        }
    }
    // store the shore texture so we can send it to our Water Material.
    shore_texture.0 = Some(pixel_data.clone());
    // create the image
    let image: Image = Image::new(
        bevy::render::render_resource::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixel_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    // make a material for debugging our shore texture
    let material = StandardMaterial {
        alpha_mode: AlphaMode::Blend,
        base_color_texture: Some(images.add(image)),
        reflectance: 0.,
        ..Default::default()
    };
    let asset_id = materials.add(material);
    material_ref.0 = Some(asset_id.clone());
    // plane
    // commands.spawn((
    //     Mesh3d(meshes.add(Mesh::from(
    //         Plane3d::default().mesh().size(WIDTH as f32, HEIGHT as f32),
    //     ))),
    //     NotShadowCaster,
    //     NotShadowReceiver,
    //     Wireframe,
    //     MeshMaterial3d(asset_id.clone()),
    //     Transform::from_xyz(0., 0., 0.),
    // ));
}
fn startup_camera(mut commands: Commands) {
    commands.spawn((
        DepthPrepass,
        DeferredPrepass,
        Camera3d::default(),
        Hdr,
        Transform::from_xyz(-2.0, 256. / 4., 256.0 / 4.).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam,
    ));
}

fn startup_light(mut commands: Commands, mut config_store: ResMut<GizmoConfigStore>) {
    let (_, light_config) = config_store.config_mut::<LightGizmoConfigGroup>();
    light_config.draw_all = true;
    light_config.color = LightGizmoColor::MatchLightColor;

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, MAGNITUDE * 1000., 0.0)
            .with_rotation(Quat::from_rotation_x(-PI / 4.)),
    ));
}
