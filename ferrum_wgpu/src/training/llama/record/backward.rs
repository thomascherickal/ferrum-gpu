use super::*;
use crate::ops::add;
use crate::ops_bwd::{colsum, embed_bwd};
use crate::ops_llama::rope;
use crate::ops_llama_bwd::{attention_gqa_bwd, rmsnorm_bwd, swiglu_bwd};

impl GpuLlamaTrainer<'_> {
    pub(super) fn record_backward(&self, dm: &Dims, ids_host: &[usize]) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, p, gr) = (self.gpu, &self.state.params, &self.state.grads);
        let (r, d, v, nl) = (dm.r, dm.d, dm.v, dm.nl);
        let (on, head) = (pi(nl, 0), pi(nl, 1));
        lin_bwd(
            g,
            (&a.head.hf, &p[head], &a.head.dlogits),
            (&a.head.dhf, &gr[head]),
            (r, d, v),
        )?;
        let norm = ((&a.x[nl], &p[on], &a.head.dhf), (&a.dx_a, &gr[on]));
        rmsnorm_bwd(g, norm.0, norm.1, &a.stats, (r, d), self.cfg.eps)?;
        let (mut cur, mut next) = (&a.dx_a, &a.dx_b);
        for l in (0..nl).rev() {
            self.backward_block(dm, l, cur, next)?;
            std::mem::swap(&mut cur, &mut next);
        }
        // Only the token-embedding gradient is wanted; the position kernel writes a scratch.
        embed_bwd(
            g,
            cur,
            ids_host,
            (&gr[0], &a.junk_pos),
            (v, d, dm.t, self.cfg.ctx),
        )
    }

    /// One block in reverse: `cur` is the gradient entering it, `next` receives the one leaving.
    fn backward_block(
        &self,
        dm: &Dims,
        l: usize,
        cur: &GpuBuf,
        next: &GpuBuf,
    ) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, c, p, gr) = (self.gpu, &self.cfg, &self.state.params, &self.state.grads);
        let (r, d, h, hd, eps) = (dm.r, dm.d, dm.h, dm.hd, c.eps);
        let lin = |k: usize, x: &GpuBuf, dy: &GpuBuf, dx: &GpuBuf, n: (usize, usize, usize)| {
            lin_bwd(g, (x, &p[pi(l, k)], dy), (dx, &gr[pi(l, k)]), n)
        };
        let norm = |k: usize, x: &GpuBuf, dy: &GpuBuf, dx: &GpuBuf| {
            let w = (x, &p[pi(l, k)], dy);
            rmsnorm_bwd(g, w, (dx, &gr[pi(l, k)]), &a.stats, (r, d), eps)
        };
        lin(DOWN, &a.act[l], cur, &a.dact, (r, h, d))?;
        swiglu_bwd(g, (&a.gate[l], &a.up[l], &a.dact), (&a.dgate, &a.dup))?;
        lin(GATE, &a.h2[l], &a.dgate, &a.dh2a, (r, d, h))?;
        lin(UP, &a.h2[l], &a.dup, &a.dh2b, (r, d, h))?;
        add(g, &a.dh2a, &a.dh2b, &a.dh2)?;
        norm(FFN_NORM, &a.x2[l], &a.dh2, &a.dx2n)?;
        add(g, cur, &a.dx2n, &a.dx2)?;
        lin(WO, &a.a[l], &a.dx2, &a.da, (r, dm.qd, d))?;
        let attn = (&a.q[l], &a.k[l], &a.v[l], &a.probs[l], &a.da);
        let shape = (dm.b, dm.t, c.heads, c.kv_heads, hd);
        attention_gqa_bwd(g, attn, (&a.dq, &a.dk, &a.dv), (&a.dot, &a.ds), shape)?;
        let back = (&a.cos, &a.neg_sin);
        rope(g, &a.dq, back, (r, c.heads, hd, c.rope_dim), c.rope_type, 0)?;
        rope(
            g,
            &a.dk,
            back,
            (r, c.kv_heads, hd, c.rope_dim),
            c.rope_type,
            0,
        )?;
        for (wi, bi, dy, dh, n) in [
            (WQ, BQ, &a.dq, &a.dh1a, dm.qd),
            (WK, BK, &a.dk, &a.dh1b, dm.kvd),
            (WV, BV, &a.dv, &a.dh1c, dm.kvd),
        ] {
            lin(wi, &a.h1[l], dy, dh, (r, d, n))?;
            if !p[pi(l, bi)].is_empty() {
                colsum(g, dy, &gr[pi(l, bi)], (r, n))?;
            }
        }
        add(g, &a.dh1a, &a.dh1b, &a.t1)?;
        add(g, &a.t1, &a.dh1c, &a.dh1)?;
        norm(ATTN_NORM, &a.x[l], &a.dh1, &a.dxn)?;
        add(g, &a.dx2, &a.dxn, next)
    }
}
