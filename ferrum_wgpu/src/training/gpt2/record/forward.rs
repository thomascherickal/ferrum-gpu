use super::{pi, Dims, GpuTrainer};
use crate::device::GpuError;
use crate::ops::{self, add};
use crate::tensor::GpuBuf;

impl GpuTrainer<'_> {
    /// `y = x * w + bias` through `tmp`, since the matmul writes its output whole.
    fn lin(
        &self,
        dm: &Dims,
        (x, w, bias): (&GpuBuf, &GpuBuf, &GpuBuf),
        (y, tmp): (&GpuBuf, &GpuBuf),
        (k, n): (usize, usize),
    ) -> Result<(), GpuError> {
        self.matmul.submit(self.gpu, x, w, tmp, (dm.r, k, n))?;
        ops::add_bias(self.gpu, tmp, bias, y)
    }

    pub(super) fn record_forward(
        &self,
        dm: &Dims,
        ids: &GpuBuf,
        targets: &GpuBuf,
    ) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, p) = (self.gpu, self.forward_weights());
        let (r, d, v, nl) = (dm.r, dm.d, dm.v, dm.nl);
        ops::embed(g, (&p[0], &p[1], ids), &a.x[0], (r, d, dm.t))?;
        for l in 0..nl {
            self.forward_block(dm, l)?;
        }
        let fb = pi(nl, 0);
        ops::layernorm(g, &a.x[nl], &p[fb], &p[fb + 1], &a.head.hf, (r, d))?;
        self.matmul
            .submit(g, &a.head.hf, &p[fb + 2], &a.head.logits, (r, d, v))?;
        let loss = (&a.head.dlogits, &a.head.row_loss, &a.head.loss);
        ops::cross_entropy(g, &a.head.logits, targets, loss, (r, v))
    }

    /// One transformer block: attention (q, k, v, o) then the feed-forward, each with a residual.
    fn forward_block(&self, dm: &Dims, l: usize) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, p) = (self.gpu, self.forward_weights());
        let (r, d, h) = (dm.r, dm.d, dm.h);
        ops::layernorm(g, &a.x[l], &p[pi(l, 0)], &p[pi(l, 1)], &a.h1[l], (r, d))?;
        for (k, y) in [(2, &a.q[l]), (4, &a.k[l]), (6, &a.v[l])] {
            let w = (&a.h1[l], &p[pi(l, k)], &p[pi(l, k + 1)]);
            self.lin(dm, w, (y, &a.tmp_c), (d, d))?;
        }
        let qkv = (&a.q[l], &a.k[l], &a.v[l]);
        ops::attention(
            g,
            qkv,
            (&a.probs[l], &a.a[l]),
            (dm.b, dm.t, self.cfg.heads, dm.hd),
        )?;
        self.lin(
            dm,
            (&a.a[l], &p[pi(l, 8)], &p[pi(l, 9)]),
            (&a.o, &a.tmp_c),
            (d, d),
        )?;
        add(g, &a.x[l], &a.o, &a.x2[l])?;
        ops::layernorm(g, &a.x2[l], &p[pi(l, 10)], &p[pi(l, 11)], &a.h2[l], (r, d))?;
        self.lin(
            dm,
            (&a.h2[l], &p[pi(l, 12)], &p[pi(l, 13)]),
            (&a.f1[l], &a.tmp_h),
            (d, h),
        )?;
        ops::gelu(g, &a.f1[l], &a.act[l])?;
        self.record_dropout(&a.act[l], l)?;
        self.lin(
            dm,
            (&a.act[l], &p[pi(l, 14)], &p[pi(l, 15)]),
            (&a.f2, &a.tmp_c),
            (h, d),
        )?;
        add(g, &a.x2[l], &a.f2, &a.x[l + 1])
    }
}
