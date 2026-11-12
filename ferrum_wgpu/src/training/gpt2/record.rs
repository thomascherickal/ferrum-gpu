//! The forward and backward dispatch sequence of one training step.

use super::GpuTrainer;
use crate::device::GpuError;
use crate::ops::{self, add};
use crate::ops_bwd as bwd;
use crate::tensor::GpuBuf;

/// Index of block `l`'s parameter `k` (0..16) in the canonical order.
fn pi(l: usize, k: usize) -> usize {
    2 + 16 * l + k
}

impl GpuTrainer<'_> {
    /// Record the forward and backward dispatches (a batch must be open).
    pub(super) fn record(
        &self,
        ids_host: &[usize],
        ids: &GpuBuf,
        targets: &GpuBuf,
    ) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, c, p, gr) = (self.gpu, &self.cfg, self.forward_weights(), &self.grads);
        let (b, t, nl) = (a.batch, a.t, c.layers);
        let (r, d, h, v, hd) = (b * t, c.dim, c.hidden, c.vocab, c.head_dim());
        let lin = |x: &GpuBuf,
                   w: &GpuBuf,
                   bias: &GpuBuf,
                   y: &GpuBuf,
                   tmp: &GpuBuf,
                   (k, n): (usize, usize)|
         -> Result<(), GpuError> {
            self.matmul.submit(g, x, w, tmp, (r, k, n))?;
            ops::add_bias(g, tmp, bias, y)
        };
        // Forward.
        ops::embed(g, (&p[0], &p[1], ids), &a.x[0], (r, d, t))?;
        for l in 0..nl {
            ops::layernorm(g, &a.x[l], &p[pi(l, 0)], &p[pi(l, 1)], &a.h1[l], (r, d))?;
            lin(
                &a.h1[l],
                &p[pi(l, 2)],
                &p[pi(l, 3)],
                &a.q[l],
                &a.tmp_c,
                (d, d),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, 4)],
                &p[pi(l, 5)],
                &a.k[l],
                &a.tmp_c,
                (d, d),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, 6)],
                &p[pi(l, 7)],
                &a.v[l],
                &a.tmp_c,
                (d, d),
            )?;
            ops::attention(
                g,
                (&a.q[l], &a.k[l], &a.v[l]),
                (&a.probs[l], &a.a[l]),
                (b, t, c.heads, hd),
            )?;
            lin(&a.a[l], &p[pi(l, 8)], &p[pi(l, 9)], &a.o, &a.tmp_c, (d, d))?;
            add(g, &a.x[l], &a.o, &a.x2[l])?;
            ops::layernorm(g, &a.x2[l], &p[pi(l, 10)], &p[pi(l, 11)], &a.h2[l], (r, d))?;
            lin(
                &a.h2[l],
                &p[pi(l, 12)],
                &p[pi(l, 13)],
                &a.f1[l],
                &a.tmp_h,
                (d, h),
            )?;
            ops::gelu(g, &a.f1[l], &a.act[l])?;
            self.record_dropout(&a.act[l], l)?;
            lin(
                &a.act[l],
                &p[pi(l, 14)],
                &p[pi(l, 15)],
                &a.f2,
                &a.tmp_c,
                (h, d),
            )?;
            add(g, &a.x2[l], &a.f2, &a.x[l + 1])?;
        }
        let fb = pi(nl, 0);
        ops::layernorm(g, &a.x[nl], &p[fb], &p[fb + 1], &a.hf, (r, d))?;
        self.matmul
            .submit(g, &a.hf, &p[fb + 2], &a.logits, (r, d, v))?;
        ops::cross_entropy(
            g,
            &a.logits,
            targets,
            (&a.dlogits, &a.row_loss, &a.loss),
            (r, v),
        )?;
        // Backward.
        bwd::linear_bwd(
            g,
            (&a.hf, &p[fb + 2], &a.dlogits),
            (&a.dhf, &gr[fb + 2], &a.junk_v),
            (r, d, v),
        )?;
        bwd::layernorm_bwd(
            g,
            (&a.x[nl], &p[fb], &a.dhf),
            (&a.dx_a, &gr[fb], &gr[fb + 1]),
            &a.stats,
            (r, d),
        )?;
        let (mut cur, mut next) = (&a.dx_a, &a.dx_b);
        for l in (0..nl).rev() {
            bwd::linear_bwd(
                g,
                (&a.act[l], &p[pi(l, 14)], cur),
                (&a.dact, &gr[pi(l, 14)], &gr[pi(l, 15)]),
                (r, h, d),
            )?;
            self.record_dropout(&a.dact, l)?;
            bwd::gelu_bwd(g, &a.dact, &a.f1[l], &a.df1)?;
            bwd::linear_bwd(
                g,
                (&a.h2[l], &p[pi(l, 12)], &a.df1),
                (&a.dh2, &gr[pi(l, 12)], &gr[pi(l, 13)]),
                (r, d, h),
            )?;
            bwd::layernorm_bwd(
                g,
                (&a.x2[l], &p[pi(l, 10)], &a.dh2),
                (&a.dx2ln, &gr[pi(l, 10)], &gr[pi(l, 11)]),
                &a.stats,
                (r, d),
            )?;
            add(g, cur, &a.dx2ln, &a.dx2)?;
            bwd::linear_bwd(
                g,
                (&a.a[l], &p[pi(l, 8)], &a.dx2),
                (&a.da, &gr[pi(l, 8)], &gr[pi(l, 9)]),
                (r, d, d),
            )?;
            bwd::attention_bwd(
                g,
                (&a.q[l], &a.k[l], &a.v[l], &a.probs[l], &a.da),
                (&a.dq, &a.dk, &a.dv),
                (&a.dot, &a.ds),
                (b, t, c.heads, hd),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 2)], &a.dq),
                (&a.dh1a, &gr[pi(l, 2)], &gr[pi(l, 3)]),
                (r, d, d),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 4)], &a.dk),
                (&a.dh1b, &gr[pi(l, 4)], &gr[pi(l, 5)]),
                (r, d, d),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 6)], &a.dv),
                (&a.dh1c, &gr[pi(l, 6)], &gr[pi(l, 7)]),
                (r, d, d),
            )?;
            add(g, &a.dh1a, &a.dh1b, &a.t1)?;
            add(g, &a.t1, &a.dh1c, &a.dh1)?;
            bwd::layernorm_bwd(
                g,
                (&a.x[l], &p[pi(l, 0)], &a.dh1),
                (&a.dxln, &gr[pi(l, 0)], &gr[pi(l, 1)]),
                &a.stats,
                (r, d),
            )?;
            add(g, &a.dx2, &a.dxln, next)?;
            std::mem::swap(&mut cur, &mut next);
        }
        bwd::embed_bwd(g, cur, ids_host, (&gr[0], &gr[1]), (v, d, t, c.ctx))
    }
}
