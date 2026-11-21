use super::*;

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
