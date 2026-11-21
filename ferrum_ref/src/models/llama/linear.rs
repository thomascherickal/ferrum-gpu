use super::*;

/// `x[rows, k] * w[k, n] (+ bias)`.
pub(super) fn linear(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    (rows, k, n): (usize, usize, usize),
) -> Vec<f32> {
    let y = matmul(x, w, rows, k, n).expect("shapes checked by construction");
    if bias.is_empty() {
        y
    } else {
        nn::add_bias(&y, bias)
    }
}

impl Llama {
    /// Logits for every position of `ids`, full causal forward pass.
    pub fn forward_logits(&self, ids: &[usize]) -> Result<Vec<f32>, String> {
        let t = ids.len();
        if t == 0 || t > self.cfg.ctx {
            return Err(format!(
                "sequence length {t} must be in 1..={}",
                self.cfg.ctx
            ));
        }
        self.run(&mut Cache::new(self.cfg.layers), ids)
    }

    /// Consume one token at position `cache.len` and return its next-token logits.
    pub fn feed(&self, cache: &mut Cache, token: usize) -> Result<Vec<f32>, String> {
        if cache.len(self.cfg.kv_dim()) >= self.cfg.ctx {
            return Err(format!("context of {} tokens is full", self.cfg.ctx));
        }
        self.run(cache, &[token])
    }

    /// Run `ids` through the stack starting at position `cache.len`, appending their keys and
    /// values to the cache; logits for every new row.
    pub(super) fn run(&self, cache: &mut Cache, ids: &[usize]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, n, kvd, qd) = (c.dim, ids.len(), c.kv_dim(), c.q_dim());
        if let Some(bad) = ids.iter().find(|&&i| i >= c.vocab) {
            return Err(format!("token {bad} is out of range for vocab {}", c.vocab));
        }
        let start = cache.len(kvd);
        let (cos, sin) = rope_tables(start, n, c.rope_dim, c.rope_base, &self.rope);
        let tables = (&cos[..], &sin[..]);
        let mut x: Vec<f32> = ids
            .iter()
            .flat_map(|&i| self.tok[i * d..(i + 1) * d].iter().copied())
            .collect();
        for (l, b) in self.blocks.iter().enumerate() {
            let h = rmsnorm(&x, &b.attn_norm, n, d, c.eps);
            let mut q = linear(&h, &b.wq, &b.bq, (n, d, qd));
            let mut k = linear(&h, &b.wk, &b.bk, (n, d, kvd));
            let v = linear(&h, &b.wv, &b.bv, (n, d, kvd));
            apply_rope(
                &mut q,
                (n, c.heads, c.head_dim, c.rope_dim),
                tables,
                c.rope_type,
            );
            apply_rope(
                &mut k,
                (n, c.kv_heads, c.head_dim, c.rope_dim),
                tables,
                c.rope_type,
            );
            cache.k[l].extend_from_slice(&k);
            cache.v[l].extend_from_slice(&v);
            let mut a = Vec::with_capacity(n * qd);
            for r in 0..n {
                a.extend(attend_row(
                    &q[r * qd..(r + 1) * qd],
                    &cache.k[l],
                    &cache.v[l],
                    start + r + 1,
                    (c.heads, c.kv_heads, c.head_dim),
                ));
            }
            x = nn::add(&x, &linear(&a, &b.wo, &[], (n, qd, d)));
            let h = rmsnorm(&x, &b.ffn_norm, n, d, c.eps);
            let g = linear(&h, &b.gate, &[], (n, d, c.hidden));
            let u = linear(&h, &b.up, &[], (n, d, c.hidden));
            x = nn::add(&x, &linear(&swiglu(&g, &u), &b.down, &[], (n, c.hidden, d)));
        }
        let h = rmsnorm(&x, &self.out_norm, n, d, c.eps);
        Ok(linear(&h, &self.head, &[], (n, d, c.vocab)))
    }
}
