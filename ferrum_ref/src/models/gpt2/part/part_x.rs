use super::*;

impl Gpt2 {
    /// Tie the output head to the token embedding: `head = tokᵀ`. A model whose head equals the
    /// transposed embedding is stored without an `output.weight` tensor (see `gpt2_gguf`).
    pub fn tie_head(&mut self) {
        self.head = crate::grad::transpose(&self.tok, self.cfg.vocab, self.cfg.dim);
    }

    /// Fold the head's gradient into the embedding's, for training a tied model: the shared
    /// matrix `W` is used as `tok = W` and `head = Wᵀ`, so `dL/dW = d_tok + d_headᵀ`. The head's
    /// own gradient is cleared (it is re-derived from `tok` after the update).
    pub fn fold_head_grad(&mut self) {
        let (v, d) = (self.cfg.vocab, self.cfg.dim);
        for t in 0..v {
            for k in 0..d {
                self.tok[t * d + k] += self.head[k * v + t];
            }
        }
        self.head.iter_mut().for_each(|x| *x = 0.0);
    }

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
    pub(crate) fn linear(
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
