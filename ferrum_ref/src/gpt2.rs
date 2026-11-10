//! GPT-2 weights (plain host f32, matrices stored `[in, out]` row-major) and the CPU oracle
//! forward pass the GPU decoder is checked against.

use crate::matmul::{matmul, rand_vec};
use crate::nn;

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

/// All weights of a GPT-2 model.
#[derive(Clone, Debug, PartialEq)]
pub struct Gpt2 {
    /// Hyperparameters.
    pub cfg: Config,
    /// Token embeddings `[vocab, dim]` (one row per token).
    pub tok: Vec<f32>,
    /// Position embeddings `[ctx, dim]`.
    pub pos: Vec<f32>,
    /// Blocks in order.
    pub blocks: Vec<Block>,
    /// Final LayerNorm gain.
    pub lnf_g: Vec<f32>,
    /// Final LayerNorm shift.
    pub lnf_b: Vec<f32>,
    /// LM head `[dim, vocab]`, no bias.
    pub head: Vec<f32>,
}

impl Gpt2 {
    /// Deterministic random weights for tests and benchmarks (small, well-conditioned).
    pub fn random(cfg: Config, seed: u64) -> Gpt2 {
        cfg.validate().expect("valid config");
        let (d, h) = (cfg.dim, cfg.hidden);
        let s = std::cell::Cell::new(seed.max(1) * 1000);
        let mat = |n: usize, scale: f32| {
            s.set(s.get() + 1);
            rand_vec(n, s.get())
                .into_iter()
                .map(|v| v * scale)
                .collect::<Vec<f32>>()
        };
        let gain = |n: usize| {
            s.set(s.get() + 1);
            rand_vec(n, s.get())
                .into_iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect::<Vec<f32>>()
        };
        let blocks = (0..cfg.layers)
            .map(|_| Block {
                ln1_g: gain(d),
                ln1_b: mat(d, 0.1),
                wq: mat(d * d, 0.5),
                bq: mat(d, 0.1),
                wk: mat(d * d, 0.5),
                bk: mat(d, 0.1),
                wv: mat(d * d, 0.5),
                bv: mat(d, 0.1),
                wo: mat(d * d, 0.5),
                bo: mat(d, 0.1),
                ln2_g: gain(d),
                ln2_b: mat(d, 0.1),
                w1: mat(d * h, 0.5),
                b1: mat(h, 0.1),
                w2: mat(h * d, 0.5),
                b2: mat(d, 0.1),
            })
            .collect();
        Gpt2 {
            cfg,
            tok: mat(cfg.vocab * d, 1.0),
            pos: mat(cfg.ctx * d, 0.5),
            blocks,
            lnf_g: gain(d),
            lnf_b: mat(d, 0.1),
            head: mat(d * cfg.vocab, 0.5),
        }
    }

    /// Row-major `[rows, n]` result of `x[rows, k] * w[k, n] + bias`.
    fn linear(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        (rows, k, n): (usize, usize, usize),
    ) -> Vec<f32> {
        let y = matmul(x, w, rows, k, n).expect("shapes checked by construction");
        match bias {
            Some(b) => nn::add_bias(&y, b),
            None => y,
        }
    }

    /// Logits for every position of `ids` (`ids.len() * vocab`), full causal forward pass.
    pub fn forward_logits(&self, ids: &[usize]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let t = ids.len();
        if t == 0 || t > c.ctx {
            return Err(format!("sequence length {t} must be in 1..={}", c.ctx));
        }
        if let Some(bad) = ids.iter().find(|&&i| i >= c.vocab) {
            return Err(format!("token {bad} is out of range for vocab {}", c.vocab));
        }
        let d = c.dim;
        let mut x = nn::embed(&self.tok, &self.pos, ids, t, d);
        for b in &self.blocks {
            let h = nn::layernorm(&x, &b.ln1_g, &b.ln1_b, t, d);
            let q = Self::linear(&h, &b.wq, Some(&b.bq), (t, d, d));
            let k = Self::linear(&h, &b.wk, Some(&b.bk), (t, d, d));
            let v = Self::linear(&h, &b.wv, Some(&b.bv), (t, d, d));
            let (a, _) = nn::attention(&q, &k, &v, (1, t, c.heads, c.head_dim()));
            x = nn::add(&x, &Self::linear(&a, &b.wo, Some(&b.bo), (t, d, d)));
            let h = nn::layernorm(&x, &b.ln2_g, &b.ln2_b, t, d);
            let f: Vec<f32> = Self::linear(&h, &b.w1, Some(&b.b1), (t, d, c.hidden))
                .into_iter()
                .map(nn::gelu)
                .collect();
            x = nn::add(&x, &Self::linear(&f, &b.w2, Some(&b.b2), (t, c.hidden, d)));
        }
        let h = nn::layernorm(&x, &self.lnf_g, &self.lnf_b, t, d);
        Ok(Self::linear(&h, &self.head, None, (t, d, c.vocab)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn tiny() -> Config {
        Config {
            vocab: 11,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 2,
            hidden: 12,
        }
    }

    #[test]
    fn config_validation() {
        assert!(tiny().validate().is_ok());
        assert!(Config { heads: 3, ..tiny() }
            .validate()
            .unwrap_err()
            .contains("divisible"));
        assert!(Config {
            layers: 0,
            ..tiny()
        }
        .validate()
        .unwrap_err()
        .contains("positive"));
        assert_eq!(tiny().head_dim(), 4);
    }

    #[test]
    fn random_weights_have_the_right_shapes_and_are_deterministic() {
        let w = Gpt2::random(tiny(), 1);
        assert_eq!(w.tok.len(), 11 * 8);
        assert_eq!(w.pos.len(), 8 * 8);
        assert_eq!(w.head.len(), 8 * 11);
        assert_eq!(w.blocks.len(), 2);
        assert_eq!(w.blocks[0].w1.len(), 8 * 12);
        assert_eq!(w, Gpt2::random(tiny(), 1));
        assert_ne!(w, Gpt2::random(tiny(), 2));
    }

    #[test]
    fn forward_is_causal_prefix_stable() {
        // Logits for the first rows must not depend on later tokens.
        let w = Gpt2::random(tiny(), 3);
        let full = w.forward_logits(&[1, 5, 2, 9, 4]).unwrap();
        let prefix = w.forward_logits(&[1, 5, 2]).unwrap();
        assert_eq!(full.len(), 5 * 11);
        assert_eq!(&full[..3 * 11], &prefix[..]);
        assert!(full.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_depends_on_the_tokens() {
        let w = Gpt2::random(tiny(), 3);
        assert_ne!(
            w.forward_logits(&[1, 2]).unwrap(),
            w.forward_logits(&[1, 3]).unwrap()
        );
    }

    #[test]
    fn forward_rejects_bad_input() {
        let w = Gpt2::random(tiny(), 3);
        assert!(w
            .forward_logits(&[])
            .unwrap_err()
            .contains("sequence length"));
        assert!(w
            .forward_logits(&[0; 9])
            .unwrap_err()
            .contains("sequence length"));
        assert!(w
            .forward_logits(&[11])
            .unwrap_err()
            .contains("out of range"));
    }
}
