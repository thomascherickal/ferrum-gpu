use super::*;

/// Element counts of the embedding, then per block `[attn_norm, wq, wk, wv, wo, ffn_norm, gate,
/// up, down]`, then the output norm and head; `None` if the config is invalid or a size
/// overflows (so a hostile config can never drive a huge allocation).
pub fn param_lens(c: &Config) -> Option<Vec<usize>> {
    c.validate().ok()?;
    let mul = |a: usize, b: usize| a.checked_mul(b);
    let (d, h) = (c.dim, c.hidden);
    let (qd, kvd) = (mul(c.heads, c.head_dim)?, mul(c.kv_heads, c.head_dim)?);
    let mut v = vec![mul(c.vocab, d)?];
    for _ in 0..c.layers {
        v.extend([d, mul(d, qd)?, mul(d, kvd)?, mul(d, kvd)?, mul(qd, d)?, d]);
        v.extend([mul(d, h)?, mul(d, h)?, mul(h, d)?]);
    }
    v.extend([d, mul(d, c.vocab)?]);
    Some(v)
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
