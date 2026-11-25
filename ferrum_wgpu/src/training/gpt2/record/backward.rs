use super::{pi, Dims, GpuTrainer};
use crate::device::GpuError;
use crate::ops::add;
use crate::ops_bwd as bwd;
use crate::tensor::GpuBuf;

impl GpuTrainer<'_> {
    pub(super) fn record_backward(&self, dm: &Dims, ids_host: &[usize]) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, p, gr) = (self.gpu, self.forward_weights(), &self.state.grads);
        let (r, d, v, nl) = (dm.r, dm.d, dm.v, dm.nl);
        let fb = pi(nl, 0);
        let head = (
            (&a.head.hf, &p[fb + 2], &a.head.dlogits),
            (&a.head.dhf, &gr[fb + 2], &a.junk_v),
        );
        bwd::linear_bwd(g, head.0, head.1, (r, d, v))?;
        let ln = (
            (&a.x[nl], &p[fb], &a.head.dhf),
            (&a.dx_a, &gr[fb], &gr[fb + 1]),
        );
        bwd::layernorm_bwd(g, ln.0, ln.1, &a.stats, (r, d))?;
        let (mut cur, mut next) = (&a.dx_a, &a.dx_b);
        for l in (0..nl).rev() {
            self.backward_block(dm, l, cur, next)?;
            std::mem::swap(&mut cur, &mut next);
        }
        bwd::embed_bwd(
            g,
            cur,
            ids_host,
            (&gr[0], &gr[1]),
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
        let (g, p, gr) = (self.gpu, self.forward_weights(), &self.state.grads);
        let (r, d, h) = (dm.r, dm.d, dm.h);
        let dims = (dm.b, dm.t, self.cfg.heads, dm.hd);
        let lin = |k: usize, x: &GpuBuf, dy: &GpuBuf, dx: &GpuBuf, n: (usize, usize, usize)| {
            let (w, dw, db) = (&p[pi(l, k)], &gr[pi(l, k)], &gr[pi(l, k + 1)]);
            bwd::linear_bwd(g, (x, w, dy), (dx, dw, db), n)
        };
        lin(14, &a.act[l], cur, &a.dact, (r, h, d))?;
        self.record_dropout(&a.dact, l)?;
        bwd::gelu_bwd(g, &a.dact, &a.f1[l], &a.df1)?;
        lin(12, &a.h2[l], &a.df1, &a.dh2, (r, d, h))?;
        let ln2 = (&a.dx2ln, &gr[pi(l, 10)], &gr[pi(l, 11)]);
        bwd::layernorm_bwd(g, (&a.x2[l], &p[pi(l, 10)], &a.dh2), ln2, &a.stats, (r, d))?;
        add(g, cur, &a.dx2ln, &a.dx2)?;
        lin(8, &a.a[l], &a.dx2, &a.da, (r, d, d))?;
        let attn = (&a.q[l], &a.k[l], &a.v[l], &a.probs[l], &a.da);
        bwd::attention_bwd(g, attn, (&a.dq, &a.dk, &a.dv), (&a.dot, &a.ds), dims)?;
        lin(2, &a.h1[l], &a.dq, &a.dh1a, (r, d, d))?;
        lin(4, &a.h1[l], &a.dk, &a.dh1b, (r, d, d))?;
        lin(6, &a.h1[l], &a.dv, &a.dh1c, (r, d, d))?;
        add(g, &a.dh1a, &a.dh1b, &a.t1)?;
        add(g, &a.t1, &a.dh1c, &a.dh1)?;
        let ln1 = (&a.dxln, &gr[pi(l, 0)], &gr[pi(l, 1)]);
        bwd::layernorm_bwd(g, (&a.x[l], &p[pi(l, 0)], &a.dh1), ln1, &a.stats, (r, d))?;
        add(g, &a.dx2, &a.dxln, next)
    }
}
