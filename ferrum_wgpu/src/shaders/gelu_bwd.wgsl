// df[i] = dact[i] * gelu'(f1[i]); exactly 0 or 1 beyond |x| = 10 like the CPU reference.
struct U { n: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> dact: array<f32>;
@group(0) @binding(1) var<storage, read> f1: array<f32>;
@group(0) @binding(2) var<storage, read_write> df: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n) { return; }
    let x = f1[g.x];
    var gr = 0.0;
    if (x >= 10.0) {
        gr = 1.0;
    } else if (x > -10.0) {
        let k = 0.7978846;
        let t = tanh(clamp(k * (x + 0.044715 * x * x * x), -20.0, 20.0));
        gr = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * x * x);
    }
    df[g.x] = dact[g.x] * gr;
}
