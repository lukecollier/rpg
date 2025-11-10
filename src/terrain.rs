use bevy::{
    app::Plugin,
    asset::{Asset, Handle},
    image::Image,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial},
    reflect::TypePath,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

pub struct TerrainPlugin;

pub type BlendMaterial = ExtendedMaterial<StandardMaterial, TextureArrayMaterial>;

pub fn create_material(
    height_map: Handle<Image>,
    ground_textures: Handle<Image>,
    ground_normals: Handle<Image>,
    slope_textures: Handle<Image>,
    slope_normals: Handle<Image>,
) -> BlendMaterial {
    let standard_material = StandardMaterial {
        opaque_render_method: bevy::pbr::OpaqueRendererMethod::Auto,
        base_color_texture: Some(height_map),
        normal_map_texture: None,
        metallic: 0.,
        reflectance: 0.,
        ..Default::default()
    };
    ExtendedMaterial {
        base: standard_material,
        extension: TextureArrayMaterial::new(
            ground_textures,
            ground_normals,
            slope_textures,
            slope_normals,
        ),
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
    // fn fragment_shader() -> ShaderRef {
    //     SHADER_ASSET_PATH.into()
    // }

    // we have to use deferred so we can add normals
    fn fragment_shader() -> ShaderRef {
        "shaders/blend_textures.wgsl".into()
    }

    // fn vertex_shader() -> ShaderRef {
    // }
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_plugins(MaterialPlugin::<BlendMaterial>::default());
    }
}
