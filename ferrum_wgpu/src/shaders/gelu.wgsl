// tanh-approximation GELU. The tanh argument is clamped to +-20 (tanh is exactly +-1 there in
// f32) so huge inputs cannot overflow inside an implementation of tanh.
struct U { n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (gid(g) >= u.n) { return; }
    let v = x[gid(g)];
    let k = 0.7978846;
    let arg = clamp(k * (v + 0.044715 * v * v * v), -20.0, 20.0);
    y[gid(g)] = 0.5 * v * (1.0 + tanh(arg));
}
