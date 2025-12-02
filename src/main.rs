use std::f32::consts::PI;

use avian3d::{
    PhysicsPlugins,
    prelude::{Collider, PhysicsDebugPlugin, RigidBody},
};
use bevy::{
    color::palettes::css::TEAL,
    core_pipeline::{
        Skybox,
        prepass::{DeferredPrepass, DepthPrepass},
    },
    dev_tools::fps_overlay::FpsOverlayPlugin,
    diagnostic::FrameTimeDiagnosticsPlugin,
    feathers::{
        FeathersPlugins,
        controls::{ButtonProps, ButtonVariant, SliderProps, button, slider},
        dark_theme::create_dark_theme,
        rounded_corners::RoundedCorners,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    light::NotShadowCaster,
    log::LogPlugin,
    pbr::{ExtendedMaterial, MaterialExtension, wireframe::WireframePlugin},
    prelude::*,
    render::{
        render_resource::{AsBindGroup, ShaderType, TextureViewDescriptor, TextureViewDimension},
        view::Hdr,
    },
    shader::ShaderRef,
    ui_widgets::{Activate, SliderPrecision, SliderStep, ValueChange, observe, slider_self_update},
};
use bevy_flycam::prelude::*;
use bevy_world_gen::terrain::{
    BlendMaterial, CHUNK_SIZE, TERRAIN_SIZE, TerrainChunkView, TerrainPlugin, TerrainSettings,
};

const WATER_SHADER_ASSET_PATH: &str = "shaders/water_material.wgsl";

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            // filter: "wgpu=trace,warn".into(),
            // level: bevy::log::Level::TRACE,
            ..Default::default()
        }))
        .add_plugins(PhysicsPlugins::default())
        .add_plugins(PhysicsDebugPlugin::default())
        .add_plugins(TerrainPlugin)
        .add_plugins(WireframePlugin::default())
        .add_plugins(FeathersPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(FpsOverlayPlugin::default())
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            speed: 100.0, // default: 12.0
            ..Default::default()
        })
        .add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, Water>>::default())
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(ShoreTexture::default())
        .insert_resource(DebugShoreTextureRef::default())
        .init_resource::<TerrainMaterials>()
        // terrain systems
        .add_systems(
            Startup,
            (
                startup_light,
                startup_ui,
                // startup_water,
                startup_camera,
            ),
        )
        .add_systems(Update, (update_skybox, debug_spawn_physics_cubes))
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

fn debug_spawn_physics_cubes(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    query: Single<&GlobalTransform, With<FlyCam>>,
) {
    if keys.just_pressed(KeyCode::Digit1) {
        let g_transform = query.into_inner();
        commands.spawn((
            RigidBody::Dynamic,
            Transform::from_translation(g_transform.translation()),
            Collider::sphere(1.),
        ));
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
                Plane3d::new(Vec3::Y, Vec2::splat(1024.))
                    .mesh()
                    .subdivisions(512),
            ),
        ),
        Transform::from_xyz(1024., 128., 1024.),
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

fn app_slider(
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    precision: i32,
    op: fn(ResMut<TerrainSettings>, f32) -> (),
) -> impl Bundle {
    (
        Node {
            width: percent(20.),
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

fn app_button(label: &str) -> impl Bundle {
    (
        Node {
            width: percent(20.),
            ..default()
        },
        children![(
            button(
                ButtonProps {
                    variant: ButtonVariant::Normal,
                    corners: RoundedCorners::All
                },
                (),
                Spawn((Text::new(label.to_string()), ThemedText))
            ),
            observe(|_activate: On<Activate>| {
                info!("Button clicked!");
            })
        )],
    )
}

fn make_slider(
    op: fn(ResMut<TerrainSettings>, f32) -> (),
) -> impl FnMut(On<ValueChange<f32>>, ResMut<TerrainSettings>) -> () {
    move |value_change, controls| {
        op(controls, value_change.value);
    }
}

#[derive(Resource, Default)]
struct LoadingSkybox {
    loading: bool,
    cubemap: Handle<Image>,
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

fn startup_ui(mut commands: Commands) {
    let defaults = TerrainSettings::default();
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
            app_slider(
                "frequency",
                defaults.frequency,
                0.1,
                1.,
                1,
                |ref mut controls, value| { controls.frequency = value }
            ),
            app_slider(
                "lacunarity",
                defaults.lacunarity,
                0.1,
                3.,
                1,
                |ref mut controls, value| { controls.lacunarity = value }
            ),
            app_slider(
                "octaves",
                defaults.octaves,
                0.,
                11.,
                0,
                |ref mut controls, value| { controls.octaves = value }
            ),
            app_slider(
                "persistence",
                defaults.persistence,
                0.0,
                1.,
                1,
                |ref mut controls, value| { controls.persistence = value }
            ),
            app_button("build")
        ],
    ));
}

fn startup_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        DepthPrepass,
        DeferredPrepass,
        Camera3d::default(),
        TerrainChunkView::new(CHUNK_SIZE, TERRAIN_SIZE),
        Hdr,
        Transform::from_xyz(-2.0, 256., 256.0 / 4.).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam,
    ));
    commands.insert_resource(LoadingSkybox {
        loading: true,
        cubemap: asset_server.load("textures/sky.png"),
    });
}

fn update_skybox(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut loading_skybox: ResMut<LoadingSkybox>,
    camera_q: Single<Entity, With<FlyCam>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !asset_server.is_loaded(&loading_skybox.cubemap) || loading_skybox.loading == false {
        return;
    }
    loading_skybox.loading = false;
    let image = images.get_mut(&loading_skybox.cubemap).unwrap();
    image.reinterpret_stacked_2d_as_array(6);
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    commands.entity(camera_q.into_inner()).insert(Skybox {
        image: loading_skybox.cubemap.clone(),
        brightness: 1000.,
        ..Default::default()
    });
}

fn startup_light(mut commands: Commands, mut config_store: ResMut<GizmoConfigStore>) {
    // ambient_light.brightness = 1.0;
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
        Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(Quat::from_rotation_x(-PI / 4.)),
    ));
}

fn rotate_light(time: Res<Time>, mut query: Query<&mut Transform, With<DirectionalLight>>) {
    for mut transform in query.iter_mut() {
        let speed = 0.3; // radians/sec
        let radius = 10.0;

        let angle = time.elapsed_secs() * speed;

        // orbit around the origin at a fixed radius
        transform.translation = Vec3::new(radius * angle.cos(), 10.0, radius * angle.sin());

        // always look at the terrain center
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
