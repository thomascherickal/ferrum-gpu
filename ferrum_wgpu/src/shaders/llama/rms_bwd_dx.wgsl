// RMSNorm backward, input gradient: with inv = (mean(x^2) + eps)^-1/2 and a = dy * g,
// dx = inv * a - x * inv^3 / n * sum(a * x). One invocation per row; serial ascending sums.
// stats[r] = inv is saved for the gain-gradient kernel.
struct U { rows: u32, cols: u32, eps: u32, p0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gain: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.rows) { return; }
    let base = r * u.cols;
    var ss = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { ss = ss + x[base + c] * x[base + c]; }
    let inv = 1.0 / sqrt(ss / f32(u.cols) + bitcast<f32>(u.eps));
    var dot = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { dot = dot + dy[base + c] * gain[c] * x[base + c]; }
    let k = inv * inv * inv * dot / f32(u.cols);
    for (var c = 0u; c < u.cols; c = c + 1u) {
        dx[base + c] = inv * dy[base + c] * gain[c] - x[base + c] * k;
    }
    stats[r] = inv;
}
