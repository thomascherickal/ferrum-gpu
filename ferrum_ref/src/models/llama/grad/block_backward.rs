use super::block_pass::Pass;
use super::*;
use crate::llama::Block;

impl Pass<'_> {
    /// One block in reverse: `dx` enters, the gradients land in `gb`, the gradient leaving is returned.
    pub(super) fn backward_block(
        &self,
        b: &Block,
        ca: &BlockCache,
        dx: &[f32],
        gb: &mut Block,
    ) -> Vec<f32> {
        let (c, rows, threads) = (&self.m.cfg, self.rows, self.threads);
        let (d, hid, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
        // x_out = x2 + down(act)
        let (dact, ddown, _) = linear_bwd(&ca.act, &b.down, dx, (rows, hid, d), threads);
        gb.down = ddown;
        let (dgate, dup) = swiglu_bwd(&ca.gate, &ca.up, &dact);
        let (dh2a, dwg, _) = linear_bwd(&ca.h2, &b.gate, &dgate, (rows, d, hid), threads);
        let (dh2b, dwu, _) = linear_bwd(&ca.h2, &b.up, &dup, (rows, d, hid), threads);
        gb.gate = dwg;
        gb.up = dwu;
        let dh2 = nn::add(&dh2a, &dh2b);
        let (dx2_n, dgn) = rmsnorm_bwd(&ca.x2, &b.ffn_norm, &dh2, (rows, d), c.eps);
        gb.ffn_norm = dgn;
        let dx2 = nn::add(dx, &dx2_n);
        // x2 = x + wo(a)
        let (da, dwo, _) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, qd, d), threads);
        gb.wo = dwo;
        let (mut dq, mut dk, dv) = gqa_bwd((&ca.q, &ca.k, &ca.v), &ca.probs, &da, self.shape());
        rope_bwd(
            &mut dq,
            rope_dims(c, rows, c.heads),
            self.tables,
            c.rope_type,
        );
        rope_bwd(
            &mut dk,
            rope_dims(c, rows, c.kv_heads),
            self.tables,
            c.rope_type,
        );
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, qd), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, kvd), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, kvd), threads);
        gb.wq = dwq;
        gb.wk = dwk;
        gb.wv = dwv;
        for (has, dst, db) in [
            (&b.bq, &mut gb.bq, dbq),
            (&b.bk, &mut gb.bk, dbk),
            (&b.bv, &mut gb.bv, dbv),
        ] {
            if !has.is_empty() {
                *dst = db;
            }
        }
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_n, dg1) = rmsnorm_bwd(&ca.x, &b.attn_norm, &dh1, (rows, d), c.eps);
        gb.attn_norm = dg1;
        nn::add(&dx2, &dx_n)
    }
}
