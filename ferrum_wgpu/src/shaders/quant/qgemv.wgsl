// y[col] = sum_k x[k] * W[col, k] for one row (decode). One workgroup of 32 invocations per output
// column: invocation e takes element e of every 32-block along k, so a column is read by 32
// invocations at once instead of one invocation walking all of k, and the 32 partial sums are
// added in a tree. Same packed layout and dequantised values as qlinear.wgsl; only the order of
// the sum differs.  kind 0 Q8_0 (9 words per record), 1 Q4_0 (5), 2 F16 (16, no scale), 3 MXFP4 (5), 4 FP8 E4M3 (9),
// 5 FP8 E5M2 (9), 6 Q2_K (22 words per super-block of 256; element e takes 8 of each).
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> red: array<f32, 32>;

//! use common/dequant

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let col = wg.x + wg.y * 65535u;
    let e = l.x;
    var acc = 0.0;
    if (col < u.n) {
        if (u.kind == 6u) {
            for (var sb = 0u; sb < u.k / 256u; sb = sb + 1u) {
                let rec = (sb * u.n + col) * 22u;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    acc = acc + x[sb * 256u + e + 32u * i] * deq_q2k(rec, e + 32u * i);
                }
            }
        } else {
            let rw = record_words(u.kind);
            for (var b = 0u; b < u.k / 32u; b = b + 1u) {
                acc = acc + x[b * 32u + e] * deq(u.kind, (b * u.n + col) * rw, e);
            }
        }
    }
    red[e] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s / 2u) {
        if (e < s) { red[e] = red[e] + red[e + s]; }
        workgroupBarrier();
    }
    if (e == 0u && col < u.n) { y[col] = red[0]; }
}
