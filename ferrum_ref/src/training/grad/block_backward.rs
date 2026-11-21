use super::block_pass::Pass;
use super::*;

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
        let (d, hid, hd) = (c.dim, c.hidden, c.head_dim());
        // x_out = x2 + f2
        let (dact, dw2, db2) = linear_bwd(&ca.act, &b.w2, dx, (rows, hid, d), threads);
        gb.w2 = dw2;
        gb.b2 = db2;
        // through the dropout mask (a dropped unit passes no gradient), then through GELU
        let df1: Vec<f32> = dact
            .iter()
            .zip(&ca.f1)
            .enumerate()
            .map(|(i, (&da, &f))| da * ca.mask.as_ref().map_or(1.0, |m| m[i]) * gelu_grad(f))
            .collect();
        let (dh2, dw1, db1) = linear_bwd(&ca.h2, &b.w1, &df1, (rows, d, hid), threads);
        gb.w1 = dw1;
        gb.b1 = db1;
        let (dx2_ln, dg2, dbl2) = layernorm_bwd(&ca.x2, &b.ln2_g, &dh2, rows, d);
        gb.ln2_g = dg2;
        gb.ln2_b = dbl2;
        let dx2 = nn::add(dx, &dx2_ln);
        // x2 = x + o
        let (da, dwo, dbo) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, d, d), threads);
        gb.wo = dwo;
        gb.bo = dbo;
        let shape = (self.batch, self.t, c.heads, hd);
        let (dq, dk, dv) = attention_bwd(&ca.q, &ca.k, &ca.v, &ca.probs, &da, shape);
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, d), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, d), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, d), threads);
        gb.wq = dwq;
        gb.bq = dbq;
        gb.wk = dwk;
        gb.bk = dbk;
        gb.wv = dwv;
        gb.bv = dbv;
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_ln, dg1, dbl1) = layernorm_bwd(&ca.x, &b.ln1_g, &dh1, rows, d);
        gb.ln1_g = dg1;
        gb.ln1_b = dbl1;
        nn::add(&dx2, &dx_ln)
    }
}
