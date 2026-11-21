use super::*;

/// Host-built RoPE tables for positions `start..start + n`: `(cos, sin)`, each `[n, rope_dim/2]`.
/// The GPU shader never calls `sin`/`cos`; it reads these.
pub fn rope_tables(
    start: usize,
    n: usize,
    rope_dim: usize,
    base: f32,
    scaling: &RopeScaling,
) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for p in start..start + n {
        for i in 0..half {
            let freq = (f64::from(base)).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
            let ff = scaling.freq_factors.get(i).copied().unwrap_or(1.0);
            let theta = p as f32 * freq / ff * (1.0 / scaling.linear_factor);
            let (s, c) = theta.sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    (cos, sin)
}

/// Rotate `rows` rows of `heads * hd` values in place; row `r` uses table row `r`. Only the first
/// `rope_dim` channels of each head rotate.
pub fn apply_rope(
    x: &mut [f32],
    (rows, heads, hd, rope_dim): (usize, usize, usize, usize),
    (cos, sin): (&[f32], &[f32]),
    ty: RopeType,
) {
    let half = rope_dim / 2;
    for r in 0..rows {
        for h in 0..heads {
            let off = (r * heads + h) * hd;
            for i in 0..half {
                let (ia, ib) = match ty {
                    RopeType::Norm => (off + 2 * i, off + 2 * i + 1),
                    RopeType::Neox => (off + i, off + i + half),
                };
                let (c, s) = (cos[r * half + i], sin[r * half + i]);
                let (a, b) = (x[ia], x[ib]);
                x[ia] = a * c - b * s;
                x[ib] = a * s + b * c;
            }
        }
    }
}

/// Grouped-query attention of one query row over `len` cached positions. `q` is `heads * hd`,
/// `k` and `v` are `[len, kv_heads * hd]`; query head `h` reads KV head `h / (heads / kv_heads)`.
pub fn attend_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    len: usize,
    (heads, kv_heads, hd): (usize, usize, usize),
) -> Vec<f32> {
    let kv_dim = kv_heads * hd;
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; heads * hd];
    let mut s = vec![0.0; len];
    for h in 0..heads {
        let off = (h / group) * hd;
        for (j, sj) in s.iter_mut().enumerate() {
            let dot: f32 = (0..hd)
                .map(|d| q[h * hd + d] * k[j * kv_dim + off + d])
                .sum();
            *sj = dot * scale;
        }
        let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for sj in s.iter_mut() {
            *sj = (*sj - max).exp();
            sum += *sj;
        }
        for d in 0..hd {
            out[h * hd + d] = (0..len).map(|j| s[j] / sum * v[j * kv_dim + off + d]).sum();
        }
    }
    out
}
