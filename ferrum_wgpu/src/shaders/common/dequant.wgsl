// Dequantise one element of a packed weight record (layout in quant/qlinear.wgsl and
// quant/qlinear_ext.wgsl); needs the kernel's `w: array<u32>` binding.
const KV = array<f32, 16>(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0);

// FP8 code `c`: kind 4 is E4M3 (3 mantissa bits, bias 120 after the f32 rebias), kind 5 is E5M2.
fn fp8(c: u32, kind: u32) -> f32 {
    let wide = kind == 4u;
    let mb = select(2u, 3u, wide);
    let e = (c >> mb) & select(31u, 15u, wide);
    let m = c & ((1u << mb) - 1u);
    var v = f32(m) * select(1.52587890625e-5, 0.001953125, wide);
    if (e != 0u) { v = bitcast<f32>(((e + select(112u, 120u, wide)) << 23u) | (m << (23u - mb))); }
    return select(v, -v, (c & 128u) != 0u);
}

// element p (0..256) of the Q2_K super-block record at word `rec` (22 words: d, dmin, 16 scale
// bytes, 64 quant bytes); the order of elements is half (128), shift (32), run (16), i
fn deq_q2k(rec: u32, p: u32) -> f32 {
    let hf = p / 128u; let sh = (p % 128u) / 32u; let l = p % 32u;
    let s = hf * 8u + sh * 2u + l / 16u;
    let sc = (w[rec + 2u + s / 4u] >> (8u * (s % 4u))) & 0xffu;
    let qi = hf * 32u + l;
    let q = ((w[rec + 6u + qi / 4u] >> (8u * (qi % 4u))) >> (2u * sh)) & 3u;
    return bitcast<f32>(w[rec]) * f32(sc & 15u) * f32(q) - bitcast<f32>(w[rec + 1u]) * f32(sc >> 4u);
}

fn record_words(kind: u32) -> u32 {
    if (kind == 1u || kind == 3u) { return 5u; }
    return select(16u, 9u, kind == 0u || (kind >= 4u && kind != 6u));
}

// element e (0..32) of the block record starting at word `rec`
fn deq(kind: u32, rec: u32, e: u32) -> f32 {
    if (kind == 2u) {
        let v = unpack2x16float(w[rec + e / 2u]);
        return select(v.y, v.x, e % 2u == 0u);
    }
    let d = bitcast<f32>(w[rec]);
    if (kind == 0u) {
        let q = bitcast<i32>(w[rec + 1u + e / 4u] << (24u - 8u * (e % 4u))) >> 24u;
        return d * f32(q);
    }
    if (kind == 1u || kind == 3u) {
        let j = e % 16u;
        let byte = (w[rec + 1u + j / 4u] >> (8u * (j % 4u))) & 0xffu;
        let nib = select(byte >> 4u, byte & 15u, e < 16u);
        return d * select(KV[nib], f32(nib) - 8.0, kind == 1u);
    }
    let code = (w[rec + 1u + e / 4u] >> (8u * (e % 4u))) & 0xffu;
    return d * fp8(code, kind);
}
