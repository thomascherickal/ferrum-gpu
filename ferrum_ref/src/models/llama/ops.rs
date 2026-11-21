// Provenance: semantics (RMSNorm, RoPE NORM/NEOX pairing and frequency scaling, GQA attention,
// SwiGLU) copied from ferrum-cpu `ferrum_core/src/llm.rs` (commit 56a9d3a). Rewritten over plain
// slices with host-built cos/sin tables so the GPU kernels can share the exact angles; ours from
// here on.
//! Forward ops of the Llama / Qwen stack on row-major f32 slices: the parity oracle for the
//! GPU kernels.

/// Which pairing RoPE rotates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    /// Interleaved adjacent pairs `(2i, 2i+1)` (llama.cpp `NORM`; the `llama` architecture).
    Norm,
    /// Split halves `(i, i + rope_dim/2)` (GPT-NeoX; the `qwen2` architecture).
    Neox,
}

/// RoPE frequency scaling as llama.cpp applies it: frequency `i` turns by
/// `pos * base^(-2i/rope_dim) / freq_factors[i] / linear_factor`.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeScaling {
    /// Per-frequency divisors (`rope_freqs.weight`, `rope_dim / 2` long); empty means none.
    pub freq_factors: Vec<f32>,
    /// Linear position divisor; 1.0 is no scaling.
    pub linear_factor: f32,
}

impl Default for RopeScaling {
    fn default() -> Self {
        RopeScaling {
            freq_factors: Vec::new(),
            linear_factor: 1.0,
        }
    }
}

impl RopeScaling {
    /// Refuse factors that cannot rotate (wrong count, zero, negative, non-finite).
    pub fn check(&self, rope_dim: usize) -> Result<(), String> {
        let n = self.freq_factors.len();
        if n != 0 && n != rope_dim / 2 {
            return Err(format!(
                "{n} frequency factors, expected rope_dim / 2 = {}",
                rope_dim / 2
            ));
        }
        let ok = |v: f32| v.is_finite() && v > 0.0;
        if !self.freq_factors.iter().all(|&v| ok(v)) || !ok(self.linear_factor) {
            return Err("RoPE scaling factors must be finite and > 0".into());
        }
        Ok(())
    }
}

/// RMSNorm per row: `y = x / sqrt(mean(x^2) + eps) * g`.
pub fn rmsnorm(x: &[f32], g: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            y[r * cols + c] = row[c] * inv * g[c];
        }
    }
    y
}

/// SiLU, `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU gate: `silu(gate) * up`, element-wise.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

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
#[cfg(test)]
mod tests;
