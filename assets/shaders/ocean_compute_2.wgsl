 // Constants
const PI: f32 = 3.14159265359;
const G: f32 = 9.81;

// Binding 0: output heightmap
@group(0) @binding(0)
var output_tex: texture_storage_2d<rgba32float, write>;

// Uniforms: simulation parameters
struct OceanParams {
    time: f32,
    peak_frequency: f32,
    wind_speed: f32,
    grid_size: u32
};
@group(0) @binding(1)
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

// Complex multiplication
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Complex exponential
fn complex_exp(angle: f32) -> vec2<f32> {
    return vec2<f32>(cos(angle), sin(angle));
}

// --- Compute shader entry point ---
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let N: u32 = params.grid_size;
    if (id.x >= N || id.y >= N) { return; }

    // Pixel coordinates
    let x = f32(id.x);
    let y = f32(id.y);

    // --- Inverse DFT sum ---
    var sum: vec2<f32> = vec2<f32>(0.0, 0.0);

    for (var kx: u32 = 0u; kx < N; kx = kx + 1u) {
        for (var ky: u32 = 0u; ky < N; ky = ky + 1u) {
            // Center wavevector
            let kxf = f32(kx) - f32(N)/2.0;
            let kyf = f32(ky) - f32(N)/2.0;
            let k = sqrt(kxf*kxf + kyf*kyf);

            // Amplitude & random phase
            let amplitude = sqrt(2.0 * jonswap_spectrum_k(k, params.peak_frequency, params.wind_speed));
            let phase = 2.0 * PI * hash22(vec2<u32>(kx, ky));
            let h0 = complex_from_angle(phase) * amplitude;

            // Time evolution
            let omega = sqrt(G * k);
            let coswt = cos(omega * params.time);
            let sinwt = sin(omega * params.time);
            let h_t = vec2<f32>(
                h0.x * coswt - h0.y * sinwt,
                h0.x * sinwt + h0.y * coswt
            );

            // Angle for inverse DFT
            let angle = 2.0 * PI * (kxf * x + kyf * y) / f32(N);
            sum += complex_mul(h_t, complex_exp(angle));
        }
    }

    // Normalize by N^2
    sum /= vec2<f32>(f32(N) * f32(N));

    // --- Scale for visualization ---
    let height = sum.x * 100.0; // tweak 10.0 to make waves visible
    let scaled = clamp(height, 0.0, 1.0);
    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(scaled, scaled, scaled, 1.0));
}

