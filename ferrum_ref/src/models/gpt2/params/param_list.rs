use super::*;

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
}
