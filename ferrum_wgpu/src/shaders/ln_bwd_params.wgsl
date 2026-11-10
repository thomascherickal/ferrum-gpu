// LayerNorm backward, gain and shift gradients: sums over rows in ascending order, one
// invocation per column, using the (mean, inv_std) stats written by ln_bwd_dx.
struct U { rows: u32, cols: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgain: array<f32>;
@group(0) @binding(4) var<storage, read_write> dshift: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = gid(g);
    if (c >= u.cols) { return; }
    var sg = 0.0;
    var sb = 0.0;
    for (var r = 0u; r < u.rows; r = r + 1u) {
        let d = dy[r * u.cols + c];
        sg = sg + d * ((x[r * u.cols + c] - stats[2u * r]) * stats[2u * r + 1u]);
        sb = sb + d;
    }
    dgain[c] = sg;
    dshift[c] = sb;
}
