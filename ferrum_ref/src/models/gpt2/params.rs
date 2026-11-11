//! Canonical parameter order, accessors, shape arithmetic and initialisation.

use super::{Block, Config, Gpt2};

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

/// Element count of every parameter in canonical order, or `None` if the config is invalid or
/// a size overflows (so a hostile config can never drive a huge allocation).
pub fn param_lens(c: &Config) -> Option<Vec<usize>> {
    c.validate().ok()?;
    let (d, h) = (c.dim, c.hidden);
    let mul = |a: usize, b: usize| a.checked_mul(b);
    let mut v = vec![mul(c.vocab, d)?, mul(c.ctx, d)?];
    for _ in 0..c.layers {
        let (dd, dh) = (mul(d, d)?, mul(d, h)?);
        v.extend([d, d, dd, d, dd, d, dd, d, dd, d, d, d, dh, h, dh, d]);
    }
    v.extend([d, d, mul(d, c.vocab)?]);
    Some(v)
}

impl Gpt2 {
    /// Build a model from parameters in canonical order; each length is checked against
    /// [`param_lens`].
    pub fn from_params(cfg: Config, mut params: Vec<Vec<f32>>) -> Result<Gpt2, String> {
        let lens = param_lens(&cfg).ok_or("invalid model config")?;
        if params.len() != lens.len() || params.iter().zip(&lens).any(|(p, &n)| p.len() != n) {
            return Err("parameter lengths do not match the config".into());
        }
        let mut it = params.drain(..);
        let mut next = || it.next().expect("count checked");
        let (tok, pos) = (next(), next());
        let blocks = (0..cfg.layers)
            .map(|_| Block {
                ln1_g: next(),
                ln1_b: next(),
                wq: next(),
                bq: next(),
                wk: next(),
                bk: next(),
                wv: next(),
                bv: next(),
                wo: next(),
                bo: next(),
                ln2_g: next(),
                ln2_b: next(),
                w1: next(),
                b1: next(),
                w2: next(),
                b2: next(),
            })
            .collect();
        Ok(Gpt2 {
            cfg,
            tok,
            pos,
            blocks,
            lnf_g: next(),
            lnf_b: next(),
            head: next(),
        })
    }

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
