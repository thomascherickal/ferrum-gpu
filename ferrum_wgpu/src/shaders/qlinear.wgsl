// y[r, col] = sum_k x[r, k] * W[col, k] with W kept packed on the device and dequantised in the
// inner loop. Blocks of 32 along k; block-major records so adjacent columns are adjacent in
// memory: record (b * n + col) holds the block's scale as f32 bits (converted from the file's
// f16 on the host, exactly) followed by the quants.
//   kind 0 (Q8_0): 9 words = scale + 32 i8, 4 per word, little-endian; value = d * f32(q).
//   kind 1 (Q4_0): 5 words = scale + 16 bytes; byte j's low nibble is element j, its high nibble
//                  element j + 16; value = d * (f32(nibble) - 8).
// The dequantised value is formed first and then multiplied by x, so it is bit-identical to the
// CPU decode; only the order of the final sum differs. One invocation per output element.
struct U { rows: u32, k: u32, n: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = gid(g);
    if (idx >= u.rows * u.n) { return; }
    let r = idx / u.n;
    let col = idx % u.n;
    let xb = r * u.k;
    var acc = 0.0;
    for (var b = 0u; b < u.k / 32u; b = b + 1u) {
        let k0 = xb + b * 32u;
        if (u.kind == 0u) {
            let rec = (b * u.n + col) * 9u;
            let d = bitcast<f32>(w[rec]);
            for (var j = 0u; j < 8u; j = j + 1u) {
                let word = w[rec + 1u + j];
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let q = bitcast<i32>(word << (24u - 8u * e)) >> 24u;
                    acc = acc + x[k0 + 4u * j + e] * (d * f32(q));
                }
            }
        } else {
            let rec = (b * u.n + col) * 5u;
            let d = bitcast<f32>(w[rec]);
            for (var j = 0u; j < 4u; j = j + 1u) {
                let word = w[rec + 1u + j];
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let byte = (word >> (8u * e)) & 0xffu;
                    acc = acc + x[k0 + 4u * j + e] * (d * (f32(byte & 15u) - 8.0));
                    acc = acc + x[k0 + 16u + 4u * j + e] * (d * (f32(byte >> 4u) - 8.0));
                }
            }
        }
    }
    y[idx] = acc;
}
