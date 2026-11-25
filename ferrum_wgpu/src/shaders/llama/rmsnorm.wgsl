// y = x / sqrt(mean(x^2) + eps) * gain, one invocation per row, serial ascending sums (the CPU
// reference order). eps arrives as f32 bits in the uniform.
struct U { rows: u32, cols: u32, eps: u32, p0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gain: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
//! use common/rms_inv

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.rows) { return; }
    let base = r * u.cols;
    let inv = rms_inv(base);
    for (var c = 0u; c < u.cols; c = c + 1u) { y[base + c] = x[base + c] * inv * gain[c]; }
}
