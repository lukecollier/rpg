#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
    pbr_functions as fns,
    pbr_bindings,
    pbr_fragment
}
#import bevy_core_pipeline::tonemapping::tone_mapping

@group(#{MATERIAL_BIND_GROUP}) @binding(0) 
var height_map : texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) 
var height_map_sampler : sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var ground_textures: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var ground_textures_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(5) var ground_normals: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var ground_normals_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(7) var slope_textures: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var slope_textures_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(9) var slope_normals: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var slope_normals_sampler: sampler;

struct TerrainMaterialParams {
    offset: vec2<f32>,
    size: vec2<f32>,
    height_mult: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(11) var<uniform> params: TerrainMaterialParams;

// Quintic fade curve and its derivative
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
fn dfade(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

// Simple gradient function based on lattice coordinates
fn hash2(p: vec2<i32>) -> f32 {
    let p2 = vec2<f32>(f32(p.x), f32(p.y));
    let h = dot(p2, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn grad2(p: vec2<i32>) -> vec2<f32> {
    let h = hash2(p) * 6.28318530718; // 2π
    return vec2<f32>(cos(h), sin(h));
}

fn simplex_noise(P: vec2<f32>) -> f32 {
    // Skew/unskew factors for 2D simplex grid
    let F2: f32 = 0.3660254037844386;  // (√3 - 1)/2
    let G2: f32 = 0.21132486540518713; // (3 - √3)/6

    // Skew to find simplex cell
    let s = (P.x + P.y) * F2;
    let i = floor(P.x + s);
    let j = floor(P.y + s);

    // Unskew to get cell origin in (x,y)
    let t = (i + j) * G2;
    let X0 = i - t;
    let Y0 = j - t;
    let x0 = P.x - X0;
    let y0 = P.y - Y0;

    // Determine which simplex triangle we are in
    var i1 = 0.0;
    var j1 = 0.0;
    if (x0 > y0) {
        i1 = 1.0;
        j1 = 0.0;
    } else {
        i1 = 0.0;
        j1 = 1.0;
    }

    // Offsets for the other two simplex corners
    let x1 = x0 - i1 + G2;
    let y1 = y0 - j1 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    // Integer lattice coordinates
    let ii = vec2<i32>(i32(i), i32(j));
    let jj = vec2<i32>(i32(i + i1), i32(j + j1));
    let kk = vec2<i32>(i32(i + 1.0), i32(j + 1.0));

    // Gradient vectors
    let g0 = grad2(ii);
    let g1 = grad2(jj);
    let g2 = grad2(kk);

    // Initialize accumulators
    var n0 = 0.0;
    var n1 = 0.0;
    var n2 = 0.0;

    // Corner 0
    let t0 = 0.5 - x0*x0 - y0*y0;
    if (t0 > 0.0) {
        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let gdot = dot(g0, vec2<f32>(x0, y0));
        n0 = t40 * gdot;
    }

    // Corner 1
    let t1 = 0.5 - x1*x1 - y1*y1;
    if (t1 > 0.0) {
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let gdot = dot(g1, vec2<f32>(x1, y1));
        n1 = t41 * gdot;
    }

    // Corner 2
    let t2 = 0.5 - x2*x2 - y2*y2;
    if (t2 > 0.0) {
        let t22 = t2 * t2;
        let t42 = t22 * t22;
        let gdot = dot(g2, vec2<f32>(x2, y2));
        n2 = t42 * gdot;
    }

    // Scale to keep range roughly [-1, 1]
    return 70.0 * (n0 + n1 + n2);
}



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

fn combine_normals_rnm(n1: vec3<f32>, n2: vec3<f32>) -> vec3<f32> {
    // Both n1 and n2 expected to be normalized, in tangent space
    let t = n1.xy * n2.z + n2.xy * n1.z;
    let z = n1.z * n2.z - dot(n1.xy, n2.xy);
    return normalize(vec3<f32>(t, z));
}

fn get_global_uv(local_uv: vec2<f32>, texture_size: vec2<f32>) -> vec2<f32> {
  let size_mult = params.size / texture_size;
  let offset_uv = params.offset / texture_size;

  return offset_uv + ((local_uv + (0.5 / params.size)) * size_mult);
}

@fragment
fn fragment(
    @builtin(front_facing) is_front: bool,
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let dims: vec2<f32> = vec2<f32>(textureDimensions(height_map));
    let uv_coords = mesh.uv;
    // Prepare a 'processed' StandardMaterial by sampling all textures to resolve
    // the material members
    var pbr_input: PbrInput = pbr_fragment::pbr_input_from_vertex_output(mesh, is_front, false);

    let height_derivs = textureSample(height_map, height_map_sampler, mesh.uv);
    let height = height_derivs.r;
    let deriv = height_derivs.gba * 2.0 - vec3<f32>(1.0);
    let normal = normalize(vec3<f32>(-deriv.x , 1.,-deriv.y));

    let raw_slope = compute_slope(normal);

    // Define a smooth transition between slope ranges
    let SLOPE_START = 0.3; // slope starts blending at ~17°
    let SLOPE_END   = 0.6; // fully cliff by ~37°

    let slope_factor = smoothstep(SLOPE_START, SLOPE_END, raw_slope);

    // we can calculate how often we should tile via the dimensions of our texture and the width of the texture array.
    // depending on the height we should change the slope index
    let tiling_factor = vec2<f32>(64.);

    let hn = clamp(height, 0.0, 1.0);

    // Define height ranges for each layer
    let h0 = 0.50;
    let h1 = 0.65;
    let h2 = 0.70;
    let h3 = 0.80;

    let blend = 0.05; // soft feather width

    var w0 = 1.0 - smoothstep(h0 - blend, h0 + blend, hn);
    var w1 = smoothstep(h0 - blend, h0 + blend, hn) * (1.0 - smoothstep(h1 - blend, h1 + blend, hn));
    var w2 = smoothstep(h1 - blend, h1 + blend, hn) * (1.0 - smoothstep(h2 - blend, h2 + blend, hn));
    var w3 = smoothstep(h2 - blend, h2 + blend, hn);

    // Normalize
    let total = w0 + w1 + w2 + w3 + 0.0001;
    w0 /= total;
    w1 /= total;
    w2 /= total;
    w3 /= total;

    var final_color = vec3(0.);
    var tex_normal = vec3(0.);
    var slope_color = vec3(0.);
    var slope_normal = vec3(0.);

    if w0 != 0. {
      let c0 = textureSample(ground_textures, ground_textures_sampler, uv_coords * tiling_factor, 0).rgb;
      let n0 = textureSample(ground_normals, ground_normals_sampler, uv_coords * tiling_factor, 0).rgb;
      let sc0 = textureSample(slope_textures, slope_textures_sampler, uv_coords * tiling_factor, 0).rgb;
      let sn0 = textureSample(slope_normals, slope_normals_sampler, uv_coords * tiling_factor, 0).rgb;
      final_color += c0 * w0;
      tex_normal += n0 * w0;
      slope_color += sc0 * w0;
      slope_normal += sn0 * w0;
    }
    if w1 != 0. {
      let c1 = textureSample(ground_textures, ground_textures_sampler, uv_coords * tiling_factor, 1).rgb;
      let n1 = textureSample(ground_normals, ground_normals_sampler, uv_coords * tiling_factor, 1).rgb;
      let sc1 = textureSample(slope_textures, slope_textures_sampler, uv_coords * tiling_factor, 1).rgb;
      let sn1 = textureSample(slope_normals, slope_normals_sampler, uv_coords * tiling_factor, 1).rgb;
      final_color += c1 * w1;
      tex_normal += n1 * w1;
      slope_color += sc1 * w1;
      slope_normal += sn1 * w1;
    }
    if w2 != 0. {
      let c2 = textureSample(ground_textures, ground_textures_sampler, uv_coords * tiling_factor, 2).rgb;
      let n2 = textureSample(ground_normals, ground_normals_sampler, uv_coords * tiling_factor, 2).rgb;
      let sc2 = textureSample(slope_textures, slope_textures_sampler, uv_coords * tiling_factor, 2).rgb;
      let sn2 = textureSample(slope_normals, slope_normals_sampler, uv_coords * tiling_factor, 2).rgb;
      final_color += c2 * w2;
      tex_normal += n2 * w2;
      slope_color += sc2 * w2;
      slope_normal += sn2 * w2;
    }
    if w3 != 0. {
      let c3 = textureSample(ground_textures, ground_textures_sampler, uv_coords * tiling_factor, 3).rgb;
      let n3 = textureSample(ground_normals, ground_normals_sampler, uv_coords * tiling_factor, 3).rgb;
      let sc3 = textureSample(slope_textures, slope_textures_sampler, uv_coords * tiling_factor, 3).rgb;
      let sn3 = textureSample(slope_normals, slope_normals_sampler, uv_coords * tiling_factor, 3).rgb;
      final_color += c3 * w3;
      tex_normal += n3 * w3;
      slope_color += sc3 * w3;
      slope_normal += sn3 * w3;
    }

    tex_normal = normalize(tex_normal * 2.0 - 1.0);
    final_color = mix(final_color, slope_color, slope_factor);
    tex_normal = mix(tex_normal, slope_normal, slope_factor);

    pbr_input.material.base_color = clamp(vec4(final_color, 1.), vec4<f32>(0.), vec4<f32>(1.));
#ifdef VERTEX_COLORS
    pbr_input.material.base_color = pbr_input.material.base_color * mesh.color;
#endif

    let double_sided = (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

    let n_merged = combine_normals_rnm(normal, tex_normal);

    pbr_input.frag_coord = mesh.position;
    pbr_input.world_position = mesh.world_position;
    pbr_input.world_normal = fns::prepare_world_normal(
        n_merged,
        double_sided,
        is_front,
    );
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;

    pbr_input.N = n_merged;

    pbr_input.material.metallic = 0.;
    pbr_input.material.perceptual_roughness = 1.;

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

