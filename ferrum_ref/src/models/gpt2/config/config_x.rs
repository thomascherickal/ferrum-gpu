/// Hyperparameters. LayerNorm epsilon is fixed at [`nn::LN_EPS`] (the kernels hard-code it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Vocabulary size.
    pub vocab: usize,
    /// Context length (rows of the position table).
    pub ctx: usize,
    /// Model width.
    pub dim: usize,
    /// Attention heads (divides `dim`).
    pub heads: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Feed-forward hidden width.
    pub hidden: usize,
}

impl Config {
    /// Width of one head.
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// Reject shapes the kernels cannot run.
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.vocab,
            self.ctx,
            self.dim,
            self.heads,
            self.layers,
            self.hidden,
        ]
        .contains(&0)
        {
            return Err("every dimension must be positive".into());
        }
        if !self.dim.is_multiple_of(self.heads) {
            return Err(format!(
                "dim {} is not divisible by heads {}",
                self.dim, self.heads
            ));
        }
        Ok(())
    }
}

/// One transformer block's parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// First LayerNorm gain.
    pub ln1_g: Vec<f32>,
    /// First LayerNorm shift.
    pub ln1_b: Vec<f32>,
    /// Query projection `[dim, dim]`.
    pub wq: Vec<f32>,
    /// Query bias.
    pub bq: Vec<f32>,
    /// Key projection `[dim, dim]`.
    pub wk: Vec<f32>,
    /// Key bias.
    pub bk: Vec<f32>,
    /// Value projection `[dim, dim]`.
    pub wv: Vec<f32>,
    /// Value bias.
    pub bv: Vec<f32>,
    /// Output projection `[dim, dim]`.
    pub wo: Vec<f32>,
    /// Output bias.
    pub bo: Vec<f32>,
    /// Second LayerNorm gain.
    pub ln2_g: Vec<f32>,
    /// Second LayerNorm shift.
    pub ln2_b: Vec<f32>,
    /// FFN up projection `[dim, hidden]`.
    pub w1: Vec<f32>,
    /// FFN up bias.
    pub b1: Vec<f32>,
    /// FFN down projection `[hidden, dim]`.
    pub w2: Vec<f32>,
    /// FFN down bias.
    pub b2: Vec<f32>,
}
