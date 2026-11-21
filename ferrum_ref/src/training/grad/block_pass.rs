use super::*;

/// What one forward/backward sweep shares: the model, the batch shape, the thread count and the
/// optional FFN-hidden dropout.
pub(super) struct Pass<'a> {
    pub(super) m: &'a Gpt2,
    pub(super) rows: usize,
    pub(super) batch: usize,
    pub(super) t: usize,
    pub(super) threads: usize,
    pub(super) drop: Option<Dropout>,
}

impl Pass<'_> {
    /// Block `bi` forward from `x`: the new activations and what the backward pass needs.
    pub(super) fn forward_block(&self, bi: usize, b: &Block, x: &[f32]) -> (Vec<f32>, BlockCache) {
        let (c, rows, threads) = (&self.m.cfg, self.rows, self.threads);
        let (d, hid, hd) = (c.dim, c.hidden, c.head_dim());
        let h1 = nn::layernorm(x, &b.ln1_g, &b.ln1_b, rows, d);
        let q = lin(&h1, &b.wq, &b.bq, (rows, d, d), threads);
        let k = lin(&h1, &b.wk, &b.bk, (rows, d, d), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, d), threads);
        let (a, probs) = nn::attention(&q, &k, &v, (self.batch, self.t, c.heads, hd));
        let x2 = nn::add(x, &lin(&a, &b.wo, &b.bo, (rows, d, d), threads));
        let h2 = nn::layernorm(&x2, &b.ln2_g, &b.ln2_b, rows, d);
        let f1 = lin(&h2, &b.w1, &b.b1, (rows, d, hid), threads);
        let act: Vec<f32> = f1.iter().map(|&v| nn::gelu(v)).collect();
        let mask = self.drop.map(|d| d.mask(bi, act.len()));
        let act: Vec<f32> = match &mask {
            Some(mk) => act.iter().zip(mk).map(|(a, m)| a * m).collect(),
            None => act,
        };
        let out = nn::add(&x2, &lin(&act, &b.w2, &b.b2, (rows, hid, d), threads));
        let cache = BlockCache {
            x: x.to_vec(),
            h1,
            q,
            k,
            v,
            probs,
            a,
            x2,
            h2,
            f1,
            act,
            mask,
        };
        (out, cache)
    }
}
