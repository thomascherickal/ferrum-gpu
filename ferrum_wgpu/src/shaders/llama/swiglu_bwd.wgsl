// act = silu(gate) * up: dgate = dact * up * s * (1 + gate * (1 - s)), dup = dact * silu(gate),
// with s = sigmoid(gate). The exponent is clamped at 80 like the forward kernel.
struct U { n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> dact: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dup: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gid(g);
    if (i >= u.n) { return; }
    let v = gate[i];
    let s = 1.0 / (1.0 + exp(min(-v, 80.0)));
    dgate[i] = dact[i] * up[i] * s * (1.0 + v * (1.0 - s));
    dup[i] = dact[i] * (v * s);
}
