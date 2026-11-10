// y[col] = sum_k x[k] * w[k, col] + bias[col]; one invocation per output, ascending k.
// Adjacent invocations read adjacent w elements, so reads coalesce.
struct U { k: u32, n: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let col = g.x;
    if (col >= u.n) { return; }
    var acc = 0.0;
    for (var p = 0u; p < u.k; p = p + 1u) {
        acc = acc + x[p] * w[p * u.n + col];
    }
    y[col] = acc + bias[col];
}
