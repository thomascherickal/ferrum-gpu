// y[col] = sum_k x[k] * w[k, col] + bias[col].
// A workgroup is 64 columns x 16 K-slices: slice s accumulates k = s, s+16, s+32, ... for its
// column, then slice 0 adds the 16 partials in slice order and the bias. Adjacent invocations
// (x) read adjacent w elements, so reads coalesce, and 16x more invocations hide memory latency.
// The summation order is fixed, so results are deterministic run to run.
struct U { k: u32, n: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

var<workgroup> part: array<f32, 1024>;

@compute @workgroup_size(64, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let col = wg.x * 64u + l.x;
    var acc = 0.0;
    if (col < u.n) {
        for (var p = l.y; p < u.k; p = p + 16u) {
            acc = acc + x[p] * w[p * u.n + col];
        }
    }
    part[l.y * 64u + l.x] = acc;
    workgroupBarrier();
    if (l.y == 0u && col < u.n) {
        var sum = part[l.x];
        for (var s = 1u; s < 16u; s = s + 1u) { sum = sum + part[s * 64u + l.x]; }
        y[col] = sum + bias[col];
    }
}
