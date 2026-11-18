// y[r, col] = sum_k x[r, k] * W[col, k] for several rows at once (prefill, scoring). Same packed
// weight layout and the same dequantised values as qlinear.wgsl (see there for the formats), but a
// 16 x 16 tile of outputs per workgroup: each step takes one 32-element block along k, dequantises
// the block of 16 columns once into workgroup memory and reuses it for 16 rows, and stages the 16
// rows of x the same way. The naive kernel dequantised every weight once per row.
// Rows of the tiles are 33 floats wide so the 16 columns read in one step hit different banks.
// The sum runs in ascending k, so it differs from the naive kernel only in order for Q4_0.
//   kind 0 Q8_0 (9 words per block record), 1 Q4_0 (5), 2 F16 (16, no scale), 3 MXFP4 (5), 4 FP8 E4M3
//   (9), 5 FP8 E5M2 (9), 6 Q2_K (22 words per super-block of 256, read element by element).
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> sx: array<f32, 528>;
var<workgroup> sw: array<f32, 528>;

const KV = array<f32, 16>(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0);

fn e4m3(c: u32) -> f32 {
    let e = (c >> 3u) & 15u;
    let m = c & 7u;
    var v = f32(m) * 0.001953125;
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    return select(v, -v, (c & 128u) != 0u);
}

fn e5m2(c: u32) -> f32 {
    let e = (c >> 2u) & 31u;
    let m = c & 3u;
    var v = f32(m) * 1.52587890625e-5;
    if (e != 0u) { v = bitcast<f32>(((e + 112u) << 23u) | (m << 21u)); }
    return select(v, -v, (c & 128u) != 0u);
}

// element p (0..256) of the Q2_K super-block record at word `rec` (22 words: d, dmin, 16 scale
// bytes, 64 quant bytes); the order of elements is half (128), shift (32), run (16), i
fn deq_q2k(rec: u32, p: u32) -> f32 {
    let hf = p / 128u;
    let r = p % 128u;
    let sh = r / 32u;
    let l = r % 32u;
    let s = hf * 8u + sh * 2u + l / 16u;
    let sc = (w[rec + 2u + s / 4u] >> (8u * (s % 4u))) & 0xffu;
    let qi = hf * 32u + l;
    let q = ((w[rec + 6u + qi / 4u] >> (8u * (qi % 4u))) >> (2u * sh)) & 3u;
    let dl = bitcast<f32>(w[rec]) * f32(sc & 15u);
    let ml = bitcast<f32>(w[rec + 1u]) * f32(sc >> 4u);
    return dl * f32(q) - ml;
}

fn record_words(kind: u32) -> u32 {
    if (kind == 0u) { return 9u; }
    if (kind == 1u || kind == 3u) { return 5u; }
    if (kind == 4u || kind == 5u) { return 9u; }
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
    if (kind == 2u) {
        let v = unpack2x16float(w[rec + e / 2u]);
        return select(v.y, v.x, e % 2u == 0u);
    }
    if (kind == 3u) {
        let d = bitcast<f32>(w[rec]);
        let j = e % 16u;
        let byte = (w[rec + 1u + j / 4u] >> (8u * (j % 4u))) & 0xffu;
        let nib = select(byte >> 4u, byte & 15u, e < 16u);
        return d * KV[nib];
    }
    if (kind >= 4u) {
        let d = bitcast<f32>(w[rec]);
        let code = (w[rec + 1u + e / 4u] >> (8u * (e % 4u))) & 0xffu;
        return d * select(e4m3(code), e5m2(code), kind == 5u);
    }
    return 0.0;
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
    let r = row0 + l.y;
    let c = col0 + l.x;
    if (r < u.rows && c < u.n) { y[r * u.n + c] = acc; }
}
