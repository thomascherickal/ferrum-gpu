// out[0] = (sum of row losses in ascending order) / rows. One invocation, fixed order.
struct U { rows: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> row_loss: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(1)
fn main() {
    var total = 0.0;
    for (var r = 0u; r < u.rows; r = r + 1u) { total = total + row_loss[r]; }
    out[0] = total * (1.0 / f32(u.rows));
}
