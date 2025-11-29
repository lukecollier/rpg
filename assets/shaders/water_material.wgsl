#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

struct WaterParams {
    amplitude : f32,
    frequency : f32,
    speed     : f32,
    phase     : f32,
    time      : f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> params : WaterParams;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    /* --- Model → World transform (Bevy helper) ------------------- */
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );


    // our weather texture could also provide wind via the red channel, where 0 is north and 255 / 2 is south;
    let wave_dir = vec2(1., 1.);
    let wave_coord = ((world_pos.x * wave_dir.x) + (world_pos.z * wave_dir.y));
    
    // this deformation will use the texture map for coasts, the idea being the "amplitude" will be multiplied by the blue channel,
    // e.g we sample the texture for ((world_pos.x, world_pos.z) / 255) * params.amplitude.
    // this way we can dictate how big the waves are. Same goes for the frequency 
    let amplitude = 0.1;
    let frequency = 1.;
    let wave = frequency * wave_coord + params.time;
    world_pos.y += amplitude * sin(wave);

    // the plan here is we take this normal and use it for our deferred normal calculation
    let dydx = amplitude * frequency * cos(wave);
    let dydz = amplitude * frequency * cos(wave);
    let normal = normalize(vec3(-dydx, 1.0, -dydz));

    /* --- Fill the required output struct ------------------------- */
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


@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // output the color directly
    return vec4(1.0,0.0,0.0, 1.0);
}
