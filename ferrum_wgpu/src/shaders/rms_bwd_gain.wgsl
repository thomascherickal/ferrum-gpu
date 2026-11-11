// RMSNorm backward, gain gradient: dgain[c] = sum_r dy[r, c] * x[r, c] * inv[r], ascending r;
// one invocation per column. `stats` holds the per-row inverse RMS from the dx kernel.
struct U { rows: u32, cols: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgain: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = gid(g);
    if (c >= u.cols) { return; }
    var s = 0.0;
    for (var r = 0u; r < u.rows; r = r + 1u) {
        s = s + dy[r * u.cols + c] * x[r * u.cols + c] * stats[r];
    }
    dgain[c] = s;
}
