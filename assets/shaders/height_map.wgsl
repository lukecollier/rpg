@group(0) @binding(0) 
var out_image : texture_storage_2d<rgba32float, write>;

@group(0) @binding(1) var<uniform> params: HeightMapParams;

struct HeightMapParams {
    frequency: f32,
    lacunarity: f32,
    octaves: i32,
    persistence: f32,
    x_from: f32,
    x_to: f32,
    y_from: f32,
    y_to: f32,
}

struct Noise2D {
    value: f32,
    deriv: vec3<f32>, // (dNdx, dNdy, 0)
};

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

fn perlin_noise(P: vec2<f32>) -> Noise2D {
    let Pi = vec2<i32>(floor(P));
    let Pf = P - floor(P);

    // Gradients and dot products for 4 lattice corners
    var g: array<vec2<f32>, 4>;
    var d: array<f32, 4>;

    let corners = array<vec2<i32>, 4>(
        vec2<i32>(0,0),
        vec2<i32>(1,0),
        vec2<i32>(0,1),
        vec2<i32>(1,1)
    );

    for (var i = 0u; i < 4u; i++) {
        let lattice = (Pi + corners[i]);
        g[i] = grad2(lattice);
        let offset = Pf - vec2<f32>(corners[i]);
        d[i] = dot(g[i], offset);
    }

    // Fade and derivative of fade
    let u = fade(Pf.x);
    let v = fade(Pf.y);
    let du = dfade(Pf.x);
    let dv = dfade(Pf.y);

    // Bilinear interpolation of noise value
    let nx0 = mix(d[0], d[1], u);
    let nx1 = mix(d[2], d[3], u);
    let nxy = mix(nx0, nx1, v);

    // Analytical derivatives
    let dNdx = mix(
        g[0].x + du * (d[1] - d[0]),
        g[2].x + du * (d[3] - d[2]),
        v
    );

    let dNdy = mix(
        g[0].y + dv * (d[2] - d[0]),
        g[1].y + dv * (d[3] - d[1]),
        u
    );

    return Noise2D(nxy, vec3<f32>(dNdx, dNdy, 0.0));
}

fn simplex_noise(P: vec2<f32>) -> Noise2D {
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
    var dndx = 0.0;
    var dndy = 0.0;

    // Corner 0
    let t0 = 0.5 - x0*x0 - y0*y0;
    if (t0 > 0.0) {
        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let gdot = dot(g0, vec2<f32>(x0, y0));
        n0 = t40 * gdot;
        let t30 = t20 * t0;
        let dtemp = -8.0 * t30 * gdot;
        dndx += dtemp * x0 + t40 * g0.x;
        dndy += dtemp * y0 + t40 * g0.y;
    }

    // Corner 1
    let t1 = 0.5 - x1*x1 - y1*y1;
    if (t1 > 0.0) {
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let gdot = dot(g1, vec2<f32>(x1, y1));
        n1 = t41 * gdot;
        let t31 = t21 * t1;
        let dtemp = -8.0 * t31 * gdot;
        dndx += dtemp * x1 + t41 * g1.x;
        dndy += dtemp * y1 + t41 * g1.y;
    }

    // Corner 2
    let t2 = 0.5 - x2*x2 - y2*y2;
    if (t2 > 0.0) {
        let t22 = t2 * t2;
        let t42 = t22 * t22;
        let gdot = dot(g2, vec2<f32>(x2, y2));
        n2 = t42 * gdot;
        let t32 = t22 * t2;
        let dtemp = -8.0 * t32 * gdot;
        dndx += dtemp * x2 + t42 * g2.x;
        dndy += dtemp * y2 + t42 * g2.y;
    }

    // Scale to keep range roughly [-1, 1]
    let value = 70.0 * (n0 + n1 + n2);
    let deriv = 70.0 * vec3<f32>(dndx, dndy, 0.0);

    return Noise2D(value, deriv);
}

fn simplex_fbm(
    p: vec2<f32>,
    octaves: i32,
    frequency: f32,
    lacunarity: f32,
    gain: f32
) -> Noise2D {
    var total: f32 = 0.0;
    var deriv: vec3<f32> = vec3<f32>(0.0);
    var amp: f32 = 1.0;
    var freq: f32 = frequency;
    var max_amp: f32 = 0.0;

    for (var i = 0; i < octaves; i = i + 1) {
        let n = simplex_noise(p * freq);

        // accumulate height
        total += n.value * amp;
        max_amp += amp;

        // accumulate derivatives (chain rule!)
        deriv += n.deriv * (amp * freq);

        // increase frequency, decrease amplitude
        freq *= lacunarity;
        amp *= gain;
    }

    // normalize
    return Noise2D(
        total / max_amp,
        deriv / max_amp
    );
}


// Writes our noise to a texture that we can use on the CPU side to handle things 
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(out_image);
    // return if the work group is outside the image size (unneccesary?)
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    // Calculate UV
    let uv = vec2<f32>(f32(id.x) / f32(dims.x - 1), f32(id.y) / f32(dims.y - 1));

    // Calculate the region we're interested in
    let x_extent = params.x_to - params.x_from;
    let y_extent = params.y_to - params.y_from;
    let coords = vec2<f32>((uv.x * x_extent) + params.x_from, (uv.y * y_extent) + params.y_from);

    let noise = simplex_fbm(coords, params.octaves, params.frequency, params.lacunarity, params.persistence);
    // lossy encoding of our values
    let noise_value = noise.value * 0.5 + 0.5;
    let noise_deriv = noise.deriv * 0.5 + 0.5;
    // Store height in r, and the derivatives in gba.
    //vec4<f32>(noise.value, noise.deriv.xyz);
    textureStore(out_image, id.xy, vec4(noise_value, noise_deriv));
}

