// y[r, col] = sum_k x[r, k] * W[col, k] for several rows at once (prefill, scoring). Same packed
// weight layout and the same dequantised values as qlinear.wgsl (see there for the formats), but a
// 16 x 16 tile of outputs per workgroup: each step takes one 32-element block along k, dequantises
// the block of 16 columns once into workgroup memory and reuses it for 16 rows, and stages the 16
// rows of x the same way. The naive kernel dequantised every weight once per row.
// Rows of the tiles are 33 floats wide so the 16 columns read in one step hit different banks.
// The sum runs in ascending k, so it differs from the naive kernel only in order for Q4_0.
//   kind 0 Q8_0 (9 words per block record), 1 Q4_0 (5), 2 F16 (16, no scale).
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> sx: array<f32, 528>;
var<workgroup> sw: array<f32, 528>;

fn record_words(kind: u32) -> u32 {
    if (kind == 0u) { return 9u; }
    if (kind == 1u) { return 5u; }
    return 16u;
}

// element e (0..32) of the block record starting at word `rec`
fn deq(kind: u32, rec: u32, e: u32) -> f32 {
    if (kind == 0u) {
        let d = bitcast<f32>(w[rec]);
        let q = bitcast<i32>(w[rec + 1u + e / 4u] << (24u - 8u * (e % 4u))) >> 24u;
        return d * f32(q);
    }
    if (kind == 1u) {
        let d = bitcast<f32>(w[rec]);
        let j = e % 16u;
        let byte = (w[rec + 1u + j / 4u] >> (8u * (j % 4u))) & 0xffu;
        let nib = select(byte >> 4u, byte & 15u, e < 16u);
        return d * (f32(nib) - 8.0);
    }
    let v = unpack2x16float(w[rec + e / 2u]);
    return select(v.y, v.x, e % 2u == 0u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let row0 = wg.y * 16u;
    let col0 = wg.x * 16u;
    let rw = record_words(u.kind);
    var acc = 0.0;
    for (var b = 0u; b < u.k / 32u; b = b + 1u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let idx = tid + s * 256u;
            let lr = idx / 32u;
            let e = idx % 32u;
            var xv = 0.0;
            if (row0 + lr < u.rows) { xv = x[(row0 + lr) * u.k + b * 32u + e]; }
            sx[lr * 33u + e] = xv;
            var wv = 0.0;
            if (col0 + lr < u.n) { wv = deq(u.kind, (b * u.n + col0 + lr) * rw, e); }
            sw[lr * 33u + e] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk = kk + 1u) {
            acc = acc + sx[l.y * 33u + kk] * sw[l.x * 33u + kk];
        }
        workgroupBarrier();
    }
    let r = row0 + l.y;
    let c = col0 + l.x;
    if (r < u.rows && c < u.n) { y[r * u.n + c] = acc; }
}
