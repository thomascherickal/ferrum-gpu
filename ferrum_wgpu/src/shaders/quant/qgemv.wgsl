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
