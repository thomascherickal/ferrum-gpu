// y = silu(gate) * up. The exponent is clamped at 80 (silu is already below 1e-33 there) so no
// implementation of exp can overflow into NaN.
struct U { n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gid(g);
    if (i >= u.n) { return; }
    let v = gate[i];
    y[i] = v / (1.0 + exp(min(-v, 80.0))) * up[i];
}
