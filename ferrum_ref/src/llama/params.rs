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
        // q/k scale with 1/sqrt(dim) so attention scores stay O(1) at any width (a saturated
        // softmax would amplify f32 rounding and make parity tests measure conditioning, not
        // kernels). Unchanged at dim 8.
        let qk = 0.5 * (8.0 / d as f32).sqrt();
        let bias = |n: usize| if biases { mat(n, 0.1) } else { Vec::new() };
        let blocks = (0..cfg.layers)
            .map(|_| Block {
                attn_norm: gain(d),
                wq: mat(d * qd, qk),
                bq: bias(qd),
                wk: mat(d * kvd, qk),
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

/// Parameters per block in canonical order (see [`Llama::params`]).
pub const BLOCK_PARAMS: usize = 12;

impl Block {
    /// The block's parameters in canonical order: attention norm, `wq`, `bq`, `wk`, `bk`, `wv`,
    /// `bv`, `wo`, FFN norm, `gate`, `up`, `down`. An absent bias is an empty vector.
    pub fn params(&self) -> [&Vec<f32>; BLOCK_PARAMS] {
        [
            &self.attn_norm,
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.wo,
            &self.ffn_norm,
            &self.gate,
            &self.up,
            &self.down,
        ]
    }

    /// Mutable access in canonical order.
    pub fn params_mut(&mut self) -> [&mut Vec<f32>; BLOCK_PARAMS] {
        [
            &mut self.attn_norm,
            &mut self.wq,
            &mut self.bq,
            &mut self.wk,
            &mut self.bk,
            &mut self.wv,
            &mut self.bv,
            &mut self.wo,
            &mut self.ffn_norm,
            &mut self.gate,
            &mut self.up,
            &mut self.down,
        ]
    }
}

impl Llama {
    /// All trainable parameters in canonical order: token embedding, each block's
    /// [`BLOCK_PARAMS`], output norm, head. (A tied head is not a separate parameter; training
    /// refuses tied models.)
    pub fn params(&self) -> Vec<&Vec<f32>> {
        let mut v = vec![&self.tok];
        self.blocks.iter().for_each(|b| v.extend(b.params()));
        v.extend([&self.out_norm, &self.head]);
        v
    }

    /// Mutable parameters in canonical order.
    pub fn params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.tok];
        self.blocks
            .iter_mut()
            .for_each(|b| v.extend(b.params_mut()));
        v.extend([&mut self.out_norm, &mut self.head]);
        v
    }

    /// A model of the same shape with every parameter zero (a gradient or moment container).
    pub fn zeros_like(&self) -> Llama {
        let mut z = self.clone();
        z.params_mut().into_iter().for_each(|p| p.fill(0.0));
        z
    }

    /// Whether AdamW decays canonical parameter `i`: weight matrices and the embedding are
    /// decayed; norm gains and biases are not.
    pub fn decays(&self, i: usize) -> bool {
        let n = self.blocks.len() * BLOCK_PARAMS;
        match i {
            0 => true,
            _ if i < 1 + n => matches!((i - 1) % BLOCK_PARAMS, 1 | 3 | 5 | 7 | 9 | 10 | 11),
            _ if i == 1 + n => false, // output norm
            _ => true,                // head
        }
    }
}
