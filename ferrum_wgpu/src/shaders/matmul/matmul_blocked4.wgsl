// 64x64 output tile per workgroup (16x16 invocations, 4x4 outputs each), K tile 16,
// ascending-k accumulation per output element. Accumulators are four named vec4 so no
// register array is indexed dynamically; A is staged transposed and both tiles are read as vec4.
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> sa: array<vec4<f32>, 256>; // [kk][64 rows] as vec4 groups of 4 rows
var<workgroup> sb: array<vec4<f32>, 256>; // [kk][64 cols] as vec4 groups of 4 cols

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    let tiles = (d.k + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        for (var s = 0u; s < 4u; s = s + 1u) {
            let idx = tid + s * 256u;
            // A element (row ar, k ak) goes to sa[ak*16 + ar/4][ar%4].
            let ar = idx / 16u;
            let ak = idx % 16u;
            var av = 0.0;
            if (row0 + ar < d.m && t * 16u + ak < d.k) { av = a[(row0 + ar) * d.k + t * 16u + ak]; }
            sa[ak * 16u + ar / 4u][ar % 4u] = av;
            // B element (k bk, col bc) goes to sb[bk*16 + bc/4][bc%4].
            let bk = idx / 64u;
            let bc = idx % 64u;
            var bv = 0.0;
            if (t * 16u + bk < d.k && col0 + bc < d.n) { bv = b[(t * 16u + bk) * d.n + col0 + bc]; }
            sb[bk * 16u + bc / 4u][bc % 4u] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ra = sa[kk * 16u + l.y];
            let rb = sb[kk * 16u + l.x];
            c0 = c0 + ra.x * rb;
            c1 = c1 + ra.y * rb;
            c2 = c2 + ra.z * rb;
            c3 = c3 + ra.w * rb;
        }
        workgroupBarrier();
    }
    let r = row0 + l.y * 4u;
    let cc = col0 + l.x * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        if (cc + j < d.n) {
            if (r < d.m) { c[r * d.n + cc + j] = c0[j]; }
            if (r + 1u < d.m) { c[(r + 1u) * d.n + cc + j] = c1[j]; }
            if (r + 2u < d.m) { c[(r + 2u) * d.n + cc + j] = c2[j]; }
            if (r + 3u < d.m) { c[(r + 3u) * d.n + cc + j] = c3[j]; }
        }
    }
}
