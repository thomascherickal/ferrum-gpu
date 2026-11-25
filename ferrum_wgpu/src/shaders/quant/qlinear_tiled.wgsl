// y[r, col] = sum_k x[r, k] * W[col, k] for several rows at once (prefill, scoring). Same packed
// layout and dequantised values as qlinear.wgsl (see there for the formats), but a 16 x 16 tile of
// outputs per workgroup: each step dequantises one 32-element block of 16 columns once into
// workgroup memory and reuses it for 16 rows (the naive kernel dequantised per row). Tile rows are
// 33 floats wide so the 16 columns read in one step hit different banks. The sum runs in
// ascending k: it differs from the naive kernel only in order for Q4_0.
//   kind 0 Q8_0 (9 words per block record), 1 Q4_0 (5), 2 F16 (16, no scale), 3 MXFP4 (5), 4 FP8 E4M3
//   (9), 5 FP8 E5M2 (9), 6 Q2_K (22 words per super-block of 256, read element by element).
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> sx: array<f32, 528>;
var<workgroup> sw: array<f32, 528>;

//! use common/dequant

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let row0 = wg.y * 16u; let col0 = wg.x * 16u;
    let rw = record_words(u.kind);
    var acc = 0.0;
    for (var b = 0u; b < u.k / 32u; b = b + 1u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let idx = tid + s * 256u;
            let lr = idx / 32u; let e = idx % 32u;
            var xv = 0.0;
            if (row0 + lr < u.rows) { xv = x[(row0 + lr) * u.k + b * 32u + e]; }
            sx[lr * 33u + e] = xv;
            var wv = 0.0;
            if (col0 + lr < u.n) {
                if (u.kind == 6u) {
                    let kb = b * 32u + e;
                    wv = deq_q2k(((kb / 256u) * u.n + col0 + lr) * 22u, kb % 256u);
                } else {
                    wv = deq(u.kind, (b * u.n + col0 + lr) * rw, e);
                }
            }
            sw[lr * 33u + e] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk = kk + 1u) {
            acc = acc + sx[l.y * 33u + kk] * sw[l.x * 33u + kk];
        }
        workgroupBarrier();
    }
    let r = row0 + l.y; let c = col0 + l.x;
    if (r < u.rows && c < u.n) { y[r * u.n + c] = acc; }
}
