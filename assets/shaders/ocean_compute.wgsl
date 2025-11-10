 // Constants
const PI: f32 = 3.14159265359;
const G: f32 = 9.81;

// Binding 0: storage texture for h0 (initial spectrum)
@group(0) @binding(0)
var h0_tex: texture_storage_2d<rg32float, write>;

// Binding 1: storage texture for h_t (time-evolved spectrum)
@group(0) @binding(1)
var h_t_tex: texture_storage_2d<rg32float, write>;

// Uniforms: simulation parameters
struct OceanParams {
    time: f32,
    peak_frequency: f32,
    wind_speed: f32,
    grid_size: u32
};
@group(0) @binding(2)
var<uniform> params: OceanParams;

// --- Utility functions ---

fn jonswap_spectrum(f: f32, f_p: f32, wind_speed: f32) -> f32 {
    if (f <= 0.0) { return 0.0; }
    let alpha = 0.076 * pow(wind_speed*wind_speed / (G*f_p), -0.22);
    let gamma = 3.3;
    let sigma = select(0.09, 0.07, f <= f_p);
    let r = f_p / f;
    let pm = alpha * G * G * pow((2.0*PI), -4.0) * pow(f, -5.0) * exp(-1.25 * pow(r, 4.0));
    let exponent = -pow(f/f_p - 1.0, 2.0) / (2.0 * sigma * sigma);
    let enhancement = pow(gamma, exp(exponent));
    return pm * enhancement;
}

fn jonswap_spectrum_k(k: f32, f_p: f32, wind_speed: f32) -> f32 {
    if (k <= 0.0) { return 0.0; }
    let f = sqrt(G * k) / (2.0 * PI);
    return jonswap_spectrum(f, f_p, wind_speed);
}

// Simple hash-based random generator
fn hash22(p: vec2<u32>) -> f32 {
    var x: u32 = p.x * 374761393u + p.y * 668265263u;
    x = (x ^ (x >> 13u)) * 1274126177u;
    return f32(x & 0x00ffffffu) / f32(0x01000000u);
}

// Generate unit complex number from angle
fn complex_from_angle(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

// --- Compute shader entry point ---
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let N: u32 = params.grid_size;
    if (id.x >= N || id.y >= N) { return; }

    // --- Compute wavevector ---
    let kx = f32(id.x) - f32(N)/2.0;
    let ky = f32(id.y) - f32(N)/2.0;
    let k = sqrt(kx*kx + ky*ky);

    // --- Generate initial amplitude using JONSWAP ---
    let amplitude = sqrt(2.0 * jonswap_spectrum_k(k, params.peak_frequency, params.wind_speed));

    // --- Generate random phase ---
    let phase = 2.0 * PI * hash22(id.xy);
    let h0_complex = complex_from_angle(phase) * amplitude;

    // --- Store initial spectrum ---
    textureStore(h0_tex, vec2<i32>(id.xy), vec4<f32>(h0_complex, 0.0, 0.0));

    // --- Time evolution ---
    let omega = sqrt(G * k);
    let coswt = cos(omega * params.time);
    let sinwt = sin(omega * params.time);
    let h_t = vec2<f32>(
        h0_complex.x * coswt - h0_complex.y * sinwt,
        h0_complex.x * sinwt + h0_complex.y * coswt
    );

    // --- Store evolved spectrum ---
    textureStore(h_t_tex, vec2<i32>(id.xy), vec4<f32>(h_t, 0.0, 0.0));
}
