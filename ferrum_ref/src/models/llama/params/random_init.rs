use super::*;

impl Llama {
    /// Deterministic random weights for tests and fixtures (small, well-conditioned). With
    /// `biases` the q, k and v projections carry a bias (the Qwen2 convention); with `tied` the
    /// head is the transposed embedding.
    pub fn random(cfg: Config, seed: u64, biases: bool, tied: bool) -> Llama {
        cfg.validate().expect("valid config");
        let (d, h) = (cfg.dim, cfg.hidden);
        let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
        let init = crate::matmul::SeededInit::new(seed);
        let mat = |n: usize, scale: f32| init.mat(n, scale);
        let gain = |n: usize| init.gain(n);
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
