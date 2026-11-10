// 64x64 output tile per workgroup (16x16 invocations, 4x4 outputs each), K tile 16,
// ascending-k accumulation per output element.
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> sa: array<f32, 1024>; // 64 rows x 16 k
var<workgroup> sb: array<f32, 1024>; // 16 k x 64 cols

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }
    let tiles = (d.k + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        for (var s = 0u; s < 4u; s = s + 1u) {
            let idx = tid + s * 256u;
            let ar = idx / 16u;
            let ak = idx % 16u;
            var av = 0.0;
            if (row0 + ar < d.m && t * 16u + ak < d.k) { av = a[(row0 + ar) * d.k + t * 16u + ak]; }
            sa[idx] = av;
            let bk = idx / 64u;
            let bc = idx % 64u;
            var bv = 0.0;
            if (t * 16u + bk < d.k && col0 + bc < d.n) { bv = b[(t * 16u + bk) * d.n + col0 + bc]; }
            sb[idx] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                ra[i] = sa[(l.y * 4u + i) * 16u + kk];
                rb[i] = sb[kk * 64u + l.x * 4u + i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + ra[i] * rb[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let r = row0 + l.y * 4u + i;
            let cc = col0 + l.x * 4u + j;
            if (r < d.m && cc < d.n) { c[r * d.n + cc] = acc[i * 4u + j]; }
        }
    }
}
