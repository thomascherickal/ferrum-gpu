//! Parameter shape arithmetic, deterministic random weights and the tied-head helper.

use super::{Block, Config, Llama, RopeScaling};
use crate::matmul::rand_vec;

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

impl Llama {
    /// Deterministic random weights for tests and fixtures (small, well-conditioned). With
    /// `biases` the q, k and v projections carry a bias (the Qwen2 convention); with `tied` the
    /// head is the transposed embedding.
    pub fn random(cfg: Config, seed: u64, biases: bool, tied: bool) -> Llama {
        cfg.validate().expect("valid config");
        let (d, h) = (cfg.dim, cfg.hidden);
        let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
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
        let bias = |n: usize| if biases { mat(n, 0.1) } else { Vec::new() };
        let blocks = (0..cfg.layers)
            .map(|_| Block {
                attn_norm: gain(d),
                wq: mat(d * qd, 0.5),
                bq: bias(qd),
                wk: mat(d * kvd, 0.5),
                bk: bias(kvd),
                wv: mat(d * kvd, 0.5),
                bv: bias(kvd),
                wo: mat(qd * d, 0.5),
                ffn_norm: gain(d),
                gate: mat(d * h, 0.5),
                up: mat(d * h, 0.5),
                down: mat(h * d, 0.5),
            })
            .collect();
        let tok = mat(cfg.vocab * d, 1.0);
        let head = if tied {
            transpose(&tok, cfg.vocab, d)
        } else {
            mat(d * cfg.vocab, 0.5)
        };
        Llama {
            cfg,
            rope: RopeScaling::default(),
            tok,
            blocks,
            out_norm: gain(d),
            head,
            tied,
        }
    }

    /// Total parameter count (biases included).
    pub fn num_params(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                [
                    &b.attn_norm,
                    &b.wq,
                    &b.bq,
                    &b.wk,
                    &b.bk,
                    &b.wv,
                    &b.bv,
                    &b.wo,
                ]
                .iter()
                .chain([&b.ffn_norm, &b.gate, &b.up, &b.down].iter())
                .map(|p| p.len())
                .sum::<usize>()
            })
            .sum();
        blocks + self.tok.len() + self.out_norm.len() + if self.tied { 0 } else { self.head.len() }
    }
}
