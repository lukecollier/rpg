#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) 
var height_map : texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) 
var height_map_sampler : sampler;

struct TerrainMaterialParams {
    offset: vec2<f32>,
    size: vec2<f32>,
    height_mult: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(11) var<uniform> params: TerrainMaterialParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var<storage, read_write> heights_buffer: array<f32>;

fn get_global_uv(local_uv: vec2<f32>, texture_size: vec2<f32>) -> vec2<f32> {
  let size_mult = params.size / texture_size;
  let offset_uv = params.offset / texture_size;

  return offset_uv + ((local_uv + (0.5 / params.size)) * size_mult);
}

@vertex
fn vertex(in: Vertex, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let dims: vec2<f32> = vec2<f32>(textureDimensions(height_map));
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );

    let uv = get_global_uv(in.uv, dims);

    let height_derivs = textureSampleLevel(height_map, height_map_sampler, uv, 0.0);
    let decode_height = height_derivs.r;
    world_pos.y += decode_height * params.height_mult;

    heights_buffer[0] = world_pos.y;

    // convert our derivs back to -1,1;
    let deriv = height_derivs.gba * 2.0 - vec3<f32>(1.0);
    let normal = normalize(vec3<f32>(-deriv.x , 1.,-deriv.y));

    var out : VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);

    /* Pass-through you may need later */
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        normal, in.instance_index
    );
    out.uv = uv;

    return out;
}

