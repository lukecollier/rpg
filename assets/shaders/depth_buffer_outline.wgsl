#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

@fragment
fn fragment(
    #ifdef MULTISAMPLED
        @builtin(sample_index) sample_index: u32,
    #endif
     mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    #ifndef MULTISAMPLED
        let sample_index = 0u;
    #endif

    var depth = bevy_pbr::prepass_utils::prepass_depth(mesh.position, sample_index);
    return vec4(depth, depth, depth, 0.);
}

