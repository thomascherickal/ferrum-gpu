// One invocation per row, serial ascending sums (same order as the CPU reference).
struct U { rows: u32, cols: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gain: array<f32>;
@group(0) @binding(2) var<storage, read> shift: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.rows) { return; }
    let base = r * u.cols;
    var sum = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { sum = sum + x[base + c]; }
    let mean = sum / f32(u.cols);
    var vs = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) {
        let d = x[base + c] - mean;
        vs = vs + d * d;
    }
    let inv = 1.0 / sqrt(vs / f32(u.cols) + 0.00001);
    for (var c = 0u; c < u.cols; c = c + 1u) {
        y[base + c] = (x[base + c] - mean) * inv * gain[c] + shift[c];
    }
}
