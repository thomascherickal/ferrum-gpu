// y[r, col] = sum_k x[r, k] * W[col, k] for the weight formats added after Q8_0, Q4_0 and F16:
// the same device layout (block-major records, record b * n + col, scale first as f32 bits
// converted exactly on the host), one invocation per output element, the dequantised value
// formed first and multiplied by x.
//   kind 3 (MXFP4): 5 words = scale + 16 bytes. The scale is the format's half scale 2^(e - 128).
//                   Byte j's low nibble is element j, its high nibble element j + 16; value =
//                   scale * KV[nibble] where KV holds twice the E2M1 values (0, 1, 2, 3, 4, 6, 8,
//                   12 and their negatives).
//   kind 4 (FP8 E4M3), kind 5 (FP8 E5M2): 9 words = scale + 32 codes, 4 per word; value = scale
//                   * the 8-bit float. The float is built from its bits (normal) or one exact
//                   power-of-two multiply (subnormal): no exp2, which GPUs may approximate.
//   kind 6 (Q2_K):  blocks of 256 elements, 22 words = d, dmin (f32 bits), 16 scale bytes, 64
//                   quant bytes. Sub-block `is` (16 of 16 elements) has scale byte sc: value =
//                   d * (sc & 15) * q - dmin * (sc >> 4) with q the 2-bit quant. Element order is
//                   half (128) -> shift (32) -> run (16) -> i, as ferrum_gguf::blocks decodes it.
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.rows * u.n) { return; }
    let col = idx % u.n;
    let xb = (idx / u.n) * u.k;
    var acc = 0.0;
    if (u.kind == 6u) {
        for (var b = 0u; b < u.k / 256u; b = b + 1u) {
            let rec = (b * u.n + col) * 22u;
            let d = bitcast<f32>(w[rec]);
            let dmin = bitcast<f32>(w[rec + 1u]);
            for (var s = 0u; s < 16u; s = s + 1u) {
                let sc = (w[rec + 2u + s / 4u] >> (8u * (s % 4u))) & 0xffu;
                let dl = d * f32(sc & 15u);
                let ml = dmin * f32(sc >> 4u);
                let hf = s / 8u;
                let sh = ((s % 8u) / 2u) * 2u;
                let run = s % 2u;
                for (var i = 0u; i < 16u; i = i + 1u) {
                    let qi = hf * 32u + run * 16u + i;
                    let byte = (w[rec + 6u + qi / 4u] >> (8u * (qi % 4u))) & 0xffu;
                    let q = (byte >> sh) & 3u;
                    let at = hf * 128u + ((s % 8u) / 2u) * 32u + run * 16u + i;
                    acc = acc + x[xb + b * 256u + at] * (dl * f32(q) - ml);
                }
            }
        }
    } else {
        for (var b = 0u; b < u.k / 32u; b = b + 1u) {
            let k0 = xb + b * 32u;
            if (u.kind == 3u) {
                let rec = (b * u.n + col) * 5u;
                let d = bitcast<f32>(w[rec]);
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let word = w[rec + 1u + j];
                    for (var e = 0u; e < 4u; e = e + 1u) {
                        let byte = (word >> (8u * e)) & 0xffu;
                        acc = acc + x[k0 + 4u * j + e] * (d * KV[byte & 15u]);
                        acc = acc + x[k0 + 16u + 4u * j + e] * (d * KV[byte >> 4u]);
                    }
                }
            } else {
                let rec = (b * u.n + col) * 9u;
                let d = bitcast<f32>(w[rec]);
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let word = w[rec + 1u + j];
                    for (var e = 0u; e < 4u; e = e + 1u) {
                        let code = (word >> (8u * e)) & 0xffu;
                        var v = e4m3(code);
                        if (u.kind == 5u) { v = e5m2(code); }
                        acc = acc + x[k0 + 4u * j + e] * (d * v);
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
