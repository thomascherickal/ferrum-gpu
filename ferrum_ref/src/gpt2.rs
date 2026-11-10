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

impl Block {
    /// The 16 parameters in canonical order.
    pub fn params(&self) -> [&Vec<f32>; 16] {
        [
            &self.ln1_g,
            &self.ln1_b,
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.wo,
            &self.bo,
            &self.ln2_g,
            &self.ln2_b,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
        ]
    }

    /// Mutable access to the 16 parameters in canonical order.
    pub fn params_mut(&mut self) -> [&mut Vec<f32>; 16] {
        [
            &mut self.ln1_g,
            &mut self.ln1_b,
            &mut self.wq,
            &mut self.bq,
            &mut self.wk,
            &mut self.bk,
            &mut self.wv,
            &mut self.bv,
            &mut self.wo,
            &mut self.bo,
            &mut self.ln2_g,
            &mut self.ln2_b,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ]
    }
}

/// True for tensors AdamW decays: weight matrices, never biases or LayerNorm parameters
/// (the canonical block order puts matrices at indices 2, 4, 6, 8, 12, 14).
pub fn is_matrix_in_block(i: usize) -> bool {
    matches!(i, 2 | 4 | 6 | 8 | 12 | 14)
}

impl Gpt2 {
    /// All parameters in canonical order: token embedding, position embedding, each block's
    /// 16 parameters, final LayerNorm gain and shift, head.
    pub fn params(&self) -> Vec<&Vec<f32>> {
        let mut v = vec![&self.tok, &self.pos];
        self.blocks.iter().for_each(|b| v.extend(b.params()));
        v.extend([&self.lnf_g, &self.lnf_b, &self.head]);
        v
    }

    /// Mutable parameters in canonical order.
    pub fn params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.tok, &mut self.pos];
        self.blocks
            .iter_mut()
            .for_each(|b| v.extend(b.params_mut()));
        v.extend([&mut self.lnf_g, &mut self.lnf_b, &mut self.head]);
        v
    }

    /// Whether the canonical parameter `i` is decayed by AdamW (weight matrices and the
    /// embedding tables are; biases and LayerNorm parameters are not).
    pub fn decays(&self, i: usize) -> bool {
        let n = self.blocks.len() * 16;
        match i {
            0 | 1 => true, // token and position embeddings are rank 2
            _ if i < 2 + n => is_matrix_in_block((i - 2) % 16),
            _ if i == 2 + n || i == 3 + n => false, // final LayerNorm
            _ => true,                              // head
        }
    }

    /// A model of the same shape with every parameter zero (a gradient or moment container).
    pub fn zeros_like(&self) -> Gpt2 {
        let mut z = self.clone();
        z.params_mut().into_iter().for_each(|p| p.fill(0.0));
        z
    }

    /// Total parameter count.
    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.len()).sum()
    }

    /// GPT-style initialisation for training from scratch: weights and embeddings ~ N(0, 0.02)
    /// (approximated by a scaled uniform with the same standard deviation), LayerNorm gains 1,
    /// every bias and shift 0.
    pub fn init(cfg: Config, seed: u64) -> Gpt2 {
        let mut m = Gpt2::random(cfg, seed);
        // uniform on [-1, 1) has std 1/sqrt(3); scale to std 0.02.
        let scale = 0.02 * 3f32.sqrt();
        let layers = cfg.layers;
        for i in 0..m.params().len() {
            let kind = match i {
                0 | 1 => 'w',
                _ if i < 2 + layers * 16 => match (i - 2) % 16 {
                    0 | 10 => 'g',
                    2 | 4 | 6 | 8 | 12 | 14 => 'w',
                    _ => 'b',
                },
                _ if i == 2 + layers * 16 => 'g',
                _ if i == 3 + layers * 16 => 'b',
                _ => 'w',
            };
            let p = &mut m.params_mut()[i];
            match kind {
                'w' => p.iter_mut().for_each(|v| *v *= scale),
                'g' => p.fill(1.0),
                _ => p.fill(0.0),
            }
        }
        m
    }
}

/// Per-layer key/value history for cached decoding (`[pos, dim]` each, grown as tokens arrive).
#[derive(Clone, Debug, Default)]
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Cache {
    /// An empty cache for a model with `layers` blocks.
    pub fn new(layers: usize) -> Self {
        Cache {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
        }
    }

    /// Tokens consumed so far.
    pub fn len(&self, dim: usize) -> usize {
        self.k.first().map_or(0, |k| k.len() / dim)
    }

    /// True when no token has been consumed.
    pub fn is_empty(&self) -> bool {
        self.k.first().is_none_or(|k| k.is_empty())
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.k
            .iter_mut()
            .chain(self.v.iter_mut())
            .for_each(Vec::clear);
    }
}

/// `y = x * w + bias` for one row, accumulating rows of `w` in ascending `k` (cache friendly).
fn gemv(x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; n];
    for (p, &xv) in x.iter().enumerate() {
        for (yv, &wv) in y.iter_mut().zip(&w[p * n..(p + 1) * n]) {
            *yv += xv * wv;
        }
    }
    if let Some(b) = bias {
        y.iter_mut().zip(b).for_each(|(v, b)| *v += b);
    }
    y
}

impl Gpt2 {
    /// Consume one token at position `cache.len` and return its next-token logits (cached decode).
    pub fn feed(&self, cache: &mut Cache, token: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, pos) = (c.dim, cache.len(c.dim));
        if pos >= c.ctx {
            return Err(format!("context of {} tokens is full", c.ctx));
        }
        if token >= c.vocab {
            return Err(format!(
                "token {token} is out of range for vocab {}",
                c.vocab
            ));
        }
        let hd = c.head_dim();
        let mut x: Vec<f32> = (0..d)
            .map(|i| self.tok[token * d + i] + self.pos[pos * d + i])
            .collect();
        for (l, b) in self.blocks.iter().enumerate() {
            let h = nn::layernorm(&x, &b.ln1_g, &b.ln1_b, 1, d);
            let q = gemv(&h, &b.wq, Some(&b.bq), d);
            cache.k[l].extend(gemv(&h, &b.wk, Some(&b.bk), d));
            cache.v[l].extend(gemv(&h, &b.wv, Some(&b.bv), d));
            let (kc, vc) = (&cache.k[l], &cache.v[l]);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut a = vec![0.0; d];
            for hh in 0..c.heads {
                let mut s: Vec<f32> = (0..=pos)
                    .map(|j| {
                        (0..hd)
                            .map(|i| q[hh * hd + i] * kc[j * d + hh * hd + i])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                s.iter_mut().for_each(|v| {
                    *v = (*v - max).exp();
                    sum += *v;
                });
                for i in 0..hd {
                    a[hh * hd + i] = (0..=pos)
                        .map(|j| s[j] / sum * vc[j * d + hh * hd + i])
                        .sum();
                }
            }
            x = nn::add(&x, &gemv(&a, &b.wo, Some(&b.bo), d));
            let h = nn::layernorm(&x, &b.ln2_g, &b.ln2_b, 1, d);
            let f: Vec<f32> = gemv(&h, &b.w1, Some(&b.b1), c.hidden)
                .into_iter()
                .map(nn::gelu)
                .collect();
            x = nn::add(&x, &gemv(&f, &b.w2, Some(&b.b2), d));
        }
        Ok(gemv(
            &nn::layernorm(&x, &self.lnf_g, &self.lnf_b, 1, d),
            &self.head,
            None,
            c.vocab,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_feed_matches_the_full_forward_at_every_position() {
        let w = Gpt2::random(tiny(), 9);
        let ids = [4usize, 0, 10, 3, 3, 7, 1, 2]; // fills the context
        let want = w.forward_logits(&ids).unwrap();
        let mut cache = Cache::new(w.cfg.layers);
        assert!(cache.is_empty());
        for (i, &t) in ids.iter().enumerate() {
            let got = w.feed(&mut cache, t).unwrap();
            let row = &want[i * 11..(i + 1) * 11];
            assert!(
                got.iter()
                    .zip(row)
                    .all(|(a, b)| (a - b).abs() < 1e-5 * b.abs().max(1.0)),
                "position {i}"
            );
        }
        assert_eq!(cache.len(8), 8);
        assert!(w.feed(&mut cache, 1).unwrap_err().contains("full"));
        cache.clear();
        assert!(cache.is_empty());
        assert!(w.feed(&mut cache, 11).unwrap_err().contains("out of range"));
    }

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
    fn params_are_in_canonical_order_and_counted() {
        let w = Gpt2::random(tiny(), 1);
        let p = w.params();
        assert_eq!(p.len(), 2 + 2 * 16 + 3);
        assert!(std::ptr::eq(p[0], &w.tok) && std::ptr::eq(p[1], &w.pos));
        assert!(std::ptr::eq(p[2], &w.blocks[0].ln1_g));
        assert!(std::ptr::eq(p[2 + 16 + 8], &w.blocks[1].wo));
        assert!(std::ptr::eq(p[p.len() - 1], &w.head));
        assert_eq!(w.num_params(), p.iter().map(|x| x.len()).sum::<usize>());
        let mut z = w.zeros_like();
        assert!(z.params().iter().all(|p| p.iter().all(|&v| v == 0.0)));
        assert_eq!(z.cfg, w.cfg);
        z.params_mut()[0][0] = 5.0;
        assert_eq!(z.tok[0], 5.0);
    }

    #[test]
    fn decay_flags_match_the_matrix_rule() {
        let w = Gpt2::random(tiny(), 1);
        let flags: Vec<bool> = (0..w.params().len()).map(|i| w.decays(i)).collect();
        assert!(flags[0] && flags[1], "embeddings decay");
        assert!(!flags[2] && !flags[3], "ln1 gain/shift do not");
        assert!(flags[4] && !flags[5], "wq decays, bq does not");
        assert!(flags[2 + 12] && !flags[2 + 13], "w1 decays, b1 does not");
        let n = flags.len();
        assert!(!flags[n - 3] && !flags[n - 2], "final LayerNorm does not");
        assert!(flags[n - 1], "head decays");
        // Cross-check against the shapes: decayed tensors are exactly the ones longer than dim.
        for (i, p) in w.params().iter().enumerate() {
            assert_eq!(
                flags[i],
                p.len() > w.cfg.hidden.max(w.cfg.dim),
                "param {i} len {}",
                p.len()
            );
        }
    }

    #[test]
    fn init_follows_the_gpt_recipe() {
        let w = Gpt2::init(tiny(), 3);
        assert!(w.blocks[0].ln1_g.iter().all(|&g| g == 1.0));
        assert!(w.lnf_g.iter().all(|&g| g == 1.0));
        assert!(w.blocks[0].bq.iter().chain(&w.lnf_b).all(|&b| b == 0.0));
        let std = (w.tok.iter().map(|v| v * v).sum::<f32>() / w.tok.len() as f32).sqrt();
        assert!((std - 0.02).abs() < 0.005, "std {std}");
        assert_eq!(w, Gpt2::init(tiny(), 3));
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
