#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) 
var height_map : texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) 
var height_map_sampler : sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(11) var<uniform> height_mult: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var<storage, read_write> heights_buffer: array<f32>;

@vertex
fn vertex(in: Vertex, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );

    let height_derivs = textureSampleLevel(height_map, height_map_sampler, in.uv, 0.0);
    // probably need to multiply by a amplitude
    let decode_height = height_derivs.r;
    // todo: Do we put this in a uniform?
    world_pos.y += decode_height * height_mult;

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
    out.uv = in.uv;

    return out;
}

