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
