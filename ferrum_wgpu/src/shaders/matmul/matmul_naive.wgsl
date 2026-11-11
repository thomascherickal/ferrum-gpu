// One invocation per output element; ascending-k accumulation.
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.y;
    let col = g.x;
    if (row >= d.m || col >= d.n) { return; }
    var acc = 0.0;
    for (var p = 0u; p < d.k; p = p + 1u) {
        acc = acc + a[row * d.k + p] * b[p * d.n + col];
    }
    c[row * d.n + col] = acc;
}
