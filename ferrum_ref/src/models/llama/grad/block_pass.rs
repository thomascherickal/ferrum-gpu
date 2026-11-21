use super::*;
use crate::llama::Block;

/// What one forward/backward sweep shares: the model, the batch shape, the thread count and the
/// RoPE tables for the whole batch.
pub(super) struct Pass<'a> {
    pub(super) m: &'a Llama,
    pub(super) rows: usize,
    pub(super) batch: usize,
    pub(super) t: usize,
    pub(super) threads: usize,
    pub(super) tables: (&'a [f32], &'a [f32]),
}

impl Pass<'_> {
    pub(super) fn shape(&self) -> (usize, usize, usize, usize, usize) {
        let c = &self.m.cfg;
        (self.batch, self.t, c.heads, c.kv_heads, c.head_dim)
    }

    /// One block forward from `x`: the new activations and everything the backward pass needs.
    pub(super) fn forward_block(&self, b: &Block, x: &[f32]) -> (Vec<f32>, BlockCache) {
        let (c, rows, threads) = (&self.m.cfg, self.rows, self.threads);
        let (d, hid, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
        let h1 = rmsnorm(x, &b.attn_norm, rows, d, c.eps);
        let mut q = lin(&h1, &b.wq, &b.bq, (rows, d, qd), threads);
        let mut k = lin(&h1, &b.wk, &b.bk, (rows, d, kvd), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, kvd), threads);
        apply_rope(
            &mut q,
            rope_dims(c, rows, c.heads),
            self.tables,
            c.rope_type,
        );
        apply_rope(
            &mut k,
            rope_dims(c, rows, c.kv_heads),
            self.tables,
            c.rope_type,
        );
        let (a, probs) = gqa_fwd(&q, &k, &v, self.shape());
        let x2 = nn::add(x, &mm(&a, &b.wo, (rows, qd, d), threads));
        let h2 = rmsnorm(&x2, &b.ffn_norm, rows, d, c.eps);
        let gate = mm(&h2, &b.gate, (rows, d, hid), threads);
        let up = mm(&h2, &b.up, (rows, d, hid), threads);
        let act = swiglu(&gate, &up);
        let out = nn::add(&x2, &mm(&act, &b.down, (rows, hid, d), threads));
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
            gate,
            up,
            act,
        };
        (out, cache)
    }
}
