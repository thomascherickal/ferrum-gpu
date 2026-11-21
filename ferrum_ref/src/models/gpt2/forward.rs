use super::*;

impl Gpt2 {
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
