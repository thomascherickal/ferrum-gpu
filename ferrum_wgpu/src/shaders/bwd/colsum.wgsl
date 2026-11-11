// out[c] = sum over rows (ascending) of x[r, c]; one invocation per column.
struct U { rows: u32, cols: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = gid(g);
    if (c >= u.cols) { return; }
    var s = 0.0;
    for (var r = 0u; r < u.rows; r = r + 1u) { s = s + x[r * u.cols + c]; }
    out[c] = s;
}
