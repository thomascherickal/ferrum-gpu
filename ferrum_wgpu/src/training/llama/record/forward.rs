use super::*;
use crate::ops::{self, add};
use crate::ops_llama::{attention_gqa, gather_rows, rmsnorm, rope, swiglu};

impl GpuLlamaTrainer<'_> {
    /// `y = x * w (+ bias)`: a bias goes through `tmp` because the matmul writes its output whole.
    fn lin(
        &self,
        dm: &Dims,
        (x, w, bias): (&GpuBuf, &GpuBuf, &GpuBuf),
        (y, tmp): (&GpuBuf, &GpuBuf),
        (k, n): (usize, usize),
    ) -> Result<(), GpuError> {
        if bias.is_empty() {
            return self.matmul.submit(self.gpu, x, w, y, (dm.r, k, n));
        }
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
        let (g, p) = (self.gpu, &self.params);
        let (r, d, v, nl) = (dm.r, dm.d, dm.v, dm.nl);
        gather_rows(g, &p[0], ids, &a.x[0], (r, d))?;
        for l in 0..nl {
            self.forward_block(dm, l)?;
        }
        let (on, head) = (pi(nl, 0), pi(nl, 1));
        rmsnorm(g, &a.x[nl], &p[on], &a.hf, (r, d), self.cfg.eps)?;
        self.matmul
            .submit(g, &a.hf, &p[head], &a.logits, (r, d, v))?;
        let loss = (&a.dlogits, &a.row_loss, &a.loss);
        ops::cross_entropy(g, &a.logits, targets, loss, (r, v))
    }

    /// One block: attention (q, k, v, rope, grouped-query, o) then SwiGLU, each with a residual.
    fn forward_block(&self, dm: &Dims, l: usize) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, c, p) = (self.gpu, &self.cfg, &self.params);
        let (r, d, h, hd, eps) = (dm.r, dm.d, dm.h, dm.hd, c.eps);
        rmsnorm(g, &a.x[l], &p[pi(l, ATTN_NORM)], &a.h1[l], (r, d), eps)?;
        let qkv = [
            (WQ, BQ, &a.q[l], &a.tmp_q, dm.qd, c.heads),
            (WK, BK, &a.k[l], &a.tmp_kv, dm.kvd, c.kv_heads),
            (WV, BV, &a.v[l], &a.tmp_kv, dm.kvd, 0),
        ];
        for (wi, bi, y, tmp, n, rope_heads) in qkv {
            let w = (&a.h1[l], &p[pi(l, wi)], &p[pi(l, bi)]);
            self.lin(dm, w, (y, tmp), (d, n))?;
            if rope_heads > 0 {
                let tables = (&a.cos, &a.sin);
                rope(
                    g,
                    y,
                    tables,
                    (r, rope_heads, hd, c.rope_dim),
                    c.rope_type,
                    0,
                )?;
            }
        }
        let shape = (dm.b, dm.t, c.heads, c.kv_heads, hd);
        let qkv = (&a.q[l], &a.k[l], &a.v[l]);
        attention_gqa(g, qkv, (&a.probs[l], &a.a[l]), shape)?;
        self.matmul
            .submit(g, &a.a[l], &p[pi(l, WO)], &a.o, (r, dm.qd, d))?;
        add(g, &a.x[l], &a.o, &a.x2[l])?;
        rmsnorm(g, &a.x2[l], &p[pi(l, FFN_NORM)], &a.h2[l], (r, d), eps)?;
        self.matmul
            .submit(g, &a.h2[l], &p[pi(l, GATE)], &a.gate[l], (r, d, h))?;
        self.matmul
            .submit(g, &a.h2[l], &p[pi(l, UP)], &a.up[l], (r, d, h))?;
        swiglu(g, &a.gate[l], &a.up[l], &a.act[l])?;
        self.matmul
            .submit(g, &a.act[l], &p[pi(l, DOWN)], &a.f2, (r, h, d))?;
        add(g, &a.x2[l], &a.f2, &a.x[l + 1])
    }
}
