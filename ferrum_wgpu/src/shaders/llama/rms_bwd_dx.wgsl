// RMSNorm backward, input gradient: with inv = (mean(x^2) + eps)^-1/2 and a = dy * g,
// dx = inv * a - x * inv^3 / n * sum(a * x). One invocation per row; serial ascending sums.
// stats[r] = inv is saved for the gain-gradient kernel.
struct U { rows: u32, cols: u32, eps: u32, p0: u32 }
//! use common/norm_bwd_io
//! use common/rms_inv

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.rows) { return; }
    let base = r * u.cols;
    let inv = rms_inv(base);
    var dot = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { dot = dot + dy[base + c] * gain[c] * x[base + c]; }
    let k = inv * inv * inv * dot / f32(u.cols);
    for (var c = 0u; c < u.cols; c = c + 1u) {
        dx[base + c] = inv * dy[base + c] * gain[c] - x[base + c] * k;
    }
    stats[r] = inv;
}
