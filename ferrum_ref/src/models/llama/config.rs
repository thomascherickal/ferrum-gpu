use super::*;

/// Hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Vocabulary size.
    pub vocab: usize,
    /// Context length.
    pub ctx: usize,
    /// Model width.
    pub dim: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads (divides `heads`).
    pub kv_heads: usize,
    /// Width of one head (`heads * head_dim` may differ from `dim`).
    pub head_dim: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// SwiGLU hidden width.
    pub hidden: usize,
    /// Rotated channels per head (even, at most `head_dim`).
    pub rope_dim: usize,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// RoPE pairing.
    pub rope_type: RopeType,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl Config {
    /// Width of the query projection output.
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Width of the key and value projection outputs.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Reject shapes the kernels cannot run.
    pub fn validate(&self) -> Result<(), String> {
        let dims = [
            self.vocab,
            self.ctx,
            self.dim,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.layers,
            self.hidden,
            self.rope_dim,
        ];
        if dims.contains(&0) {
            return Err("every dimension must be positive".into());
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return Err(format!(
                "heads {} is not a multiple of kv_heads {}",
                self.heads, self.kv_heads
            ));
        }
        if self.rope_dim > self.head_dim || !self.rope_dim.is_multiple_of(2) {
            return Err(format!(
                "rope_dim {} must be even and at most head_dim {}",
                self.rope_dim, self.head_dim
            ));
        }
        let bad = |v: f32| !v.is_finite() || v <= 0.0;
        if bad(self.eps) || bad(self.rope_base) {
            return Err("eps and rope_base must be finite and positive".into());
        }
        Ok(())
    }
}
