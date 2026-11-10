// 16x16 shared-memory tiles, one output per invocation, ascending-k accumulation.
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row = g.y;
    let col = g.x;
    var acc = 0.0;
    let tiles = (d.k + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ak = t * 16u + l.x;
        var av = 0.0;
        if (row < d.m && ak < d.k) { av = a[row * d.k + ak]; }
        ta[l.y * 16u + l.x] = av;
        let bk = t * 16u + l.y;
        var bv = 0.0;
        if (bk < d.k && col < d.n) { bv = b[bk * d.n + col]; }
        tb[l.y * 16u + l.x] = bv;
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + ta[l.y * 16u + i] * tb[i * 16u + l.x];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { c[row * d.n + col] = acc; }
}
