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
        let init = crate::matmul::SeededInit::new(seed);
        let mat = |n: usize, scale: f32| init.mat(n, scale);
        let gain = |n: usize| init.gain(n);
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
}
