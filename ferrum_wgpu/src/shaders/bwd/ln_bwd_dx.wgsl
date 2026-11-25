// LayerNorm backward, input gradient. One invocation per row; also stores (mean, inv_std)
// per row for the parameter-gradient kernel. Same formulas and sum order as the CPU reference.
struct U { rows: u32, cols: u32, p1: u32, p2: u32 }
//! use common/norm_bwd_io

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = gid(g);
    if (r >= u.rows) { return; }
    let base = r * u.cols;
    let n = f32(u.cols);
    var sum = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) { sum = sum + x[base + c]; }
    let mean = sum / n;
    var vs = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) {
        let dd = x[base + c] - mean;
        vs = vs + dd * dd;
    }
    let inv = 1.0 / sqrt(vs / n + 0.00001);
    stats[2u * r] = mean;
    stats[2u * r + 1u] = inv;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = 0u; c < u.cols; c = c + 1u) {
        let dxh = dy[base + c] * gain[c];
        s1 = s1 + dxh;
        s2 = s2 + dxh * ((x[base + c] - mean) * inv);
    }
    s1 = s1 / n;
    s2 = s2 / n;
    for (var c = 0u; c < u.cols; c = c + 1u) {
        let xhat = (x[base + c] - mean) * inv;
        dx[base + c] = inv * (dy[base + c] * gain[c] - s1 - xhat * s2);
    }
}
