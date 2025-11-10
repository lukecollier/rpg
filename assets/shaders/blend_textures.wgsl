#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
    pbr_functions as fns,
    pbr_bindings,
    pbr_fragment
}
#import bevy_core_pipeline::tonemapping::tone_mapping

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var ground_textures: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var ground_textures_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(102) var ground_normals: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var ground_normals_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(104) var slope_textures: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var slope_textures_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(106) var slope_normals: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var slope_normals_sampler: sampler;

fn unpack_normal(n: vec4<f32>) -> vec3<f32> {
  return normalize(n.xyz * 2.0 - vec3<f32>(1.0));
}

fn compute_slope(normal: vec3<f32>) -> f32 {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    // Ensure the normal is normalized
    let n = normalize(normal);
    // 0 = flat, 1 = vertical
    var slope = 1.0 - dot(n, up);
    // Optional clamp, if you want to prevent weird values for overhangs
    slope = clamp(slope, 0.0, 1.0);
    return slope;
}

@fragment
fn fragment(
    @builtin(front_facing) is_front: bool,
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
  let uv_coords = mesh.uv * vec2<f32>(32., 32.);
    // to get the layer we blend through the r,b,g,a (4 textures max)
    // todo: We will need to make this a deferred shader and update the normals depending on the 

    // Prepare a 'processed' StandardMaterial by sampling all textures to resolve
    // the material members
    var pbr_input: PbrInput = pbr_fragment::pbr_input_from_standard_material(mesh, is_front);

    let raw_slope = compute_slope(mesh.world_normal);

    // Define a smooth transition between slope ranges
    let SLOPE_START = 0.2; // slope starts blending at ~17°
    let SLOPE_END   = 0.6; // fully cliff by ~37°

    let slope_factor = smoothstep(SLOPE_START, SLOPE_END, raw_slope);

    let mask_color = textureSample(
        pbr_bindings::base_color_texture,
        pbr_bindings::base_color_sampler,
        mesh.uv
        );

    // Calculate the max values in our mask texture
    // todo: The mask need's to be a value from 0-1 (representing our heights)
    let color_sum = mask_color.r + mask_color.g + mask_color.b + mask_color.a + 1e-6;
    
    // Calculate how much we should blend each layer by
    let weights = mask_color / color_sum;

    // red channel
    var final_color = vec4<f32>(0.);
    final_color += textureSample(ground_textures, ground_textures_sampler, uv_coords, i32(0)) * weights.r;
    final_color += textureSample(ground_textures, ground_textures_sampler, uv_coords, i32(1)) * weights.g;
    final_color += textureSample(ground_textures, ground_textures_sampler, uv_coords, i32(2)) * weights.b;
    final_color += textureSample(ground_textures, ground_textures_sampler, uv_coords, i32(3)) * weights.a;

    // depending on the height we should change the slope index
    let slope_color = textureSample(slope_textures, slope_textures_sampler, uv_coords, i32(1));
    final_color = mix(final_color, slope_color, slope_factor);

    pbr_input.material.base_color = clamp(final_color, vec4<f32>(0.), vec4<f32>(1.));
#ifdef VERTEX_COLORS
    pbr_input.material.base_color = pbr_input.material.base_color * mesh.color;
#endif

    let double_sided = (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

    let vertex_N = normalize(mesh.world_normal);

    var tex_N = vec3<f32>(0.);
    tex_N += unpack_normal(textureSample(ground_normals, ground_normals_sampler, uv_coords, i32(0))) * weights.r;
    tex_N += unpack_normal(textureSample(ground_normals, ground_normals_sampler, uv_coords, i32(1))) * weights.g;
    tex_N += unpack_normal(textureSample(ground_normals, ground_normals_sampler, uv_coords, i32(2))) * weights.b;
    tex_N += unpack_normal(textureSample(ground_normals, ground_normals_sampler, uv_coords, i32(3))) * weights.a;

    // depending on the height we should change the slope index
    let slope_normal = unpack_normal(textureSample(slope_normals, slope_normals_sampler, uv_coords, i32(0)));
    tex_N = mix(tex_N, slope_normal, slope_factor);

    let final_N = normalize(mix(vertex_N, tex_N, 0.5));

    pbr_input.frag_coord = mesh.position;
    pbr_input.world_position = mesh.world_position;
    pbr_input.world_normal = fns::prepare_world_normal(
        final_N,
        double_sided,
        is_front,
    );

    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;

    pbr_input.N = normalize(pbr_input.world_normal);

#ifdef VERTEX_TANGENTS
    let Nt = textureSampleBias(pbr_bindings::normal_map_texture, pbr_bindings::normal_map_sampler, mesh.uv, view.mip_bias).rgb;
    let TBN = fns::calculate_tbn_mikktspace(mesh.world_normal, mesh.world_tangent);
    pbr_input.N = fns::apply_normal_mapping(
        pbr_input.material.flags,
        TBN,
        double_sided,
        is_front,
        Nt,
    );
#endif

    pbr_input.V = fns::calculate_view(mesh.world_position, pbr_input.is_orthographic);

    return tone_mapping(fns::apply_pbr_lighting(pbr_input), view.color_grading);
}

