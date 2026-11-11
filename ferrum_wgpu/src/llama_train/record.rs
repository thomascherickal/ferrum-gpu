//! The forward and backward dispatch sequence of one Llama training step.

use super::GpuLlamaTrainer;
use crate::device::{Gpu, GpuError};
use crate::ops::{self, add};
use crate::ops_bwd::{colsum, embed_bwd, matmul_ex};
use crate::ops_llama::{attention_gqa, gather_rows, rmsnorm, rope, swiglu};
use crate::ops_llama_bwd::{attention_gqa_bwd, rmsnorm_bwd, swiglu_bwd};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::BLOCK_PARAMS;

/// Index of block `l`'s parameter `k` (0..12) in the canonical order.
fn pi(l: usize, k: usize) -> usize {
    1 + BLOCK_PARAMS * l + k
}

const ATTN_NORM: usize = 0;
const WQ: usize = 1;
const BQ: usize = 2;
const WK: usize = 3;
const BK: usize = 4;
const WV: usize = 5;
const BV: usize = 6;
const WO: usize = 7;
const FFN_NORM: usize = 8;
const GATE: usize = 9;
const UP: usize = 10;
const DOWN: usize = 11;

/// `dx = dy * w^T` and `dw = x^T * dy` for `y = x * w` (`x` `[m,k]`, `w` `[k,n]`).
fn lin_bwd(
    gpu: &Gpu,
    (x, w, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dw): (&GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
) -> Result<(), GpuError> {
    matmul_ex(gpu, (dy, w, dx), (m, n, k), (false, true))?;
    matmul_ex(gpu, (x, dy, dw), (k, m, n), (true, false))
}

impl GpuLlamaTrainer<'_> {
    /// Record the forward and backward dispatches (a batch must be open).
    pub(super) fn record(
        &self,
        ids_host: &[usize],
        ids: &GpuBuf,
        targets: &GpuBuf,
    ) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, c, p, gr) = (self.gpu, &self.cfg, &self.params, &self.grads);
        let (b, t, nl) = (a.batch, a.t, c.layers);
        let (r, d, h, v, hd) = (b * t, c.dim, c.hidden, c.vocab, c.head_dim);
        let (qd, kvd, eps, ty, rd) = (c.q_dim(), c.kv_dim(), c.eps, c.rope_type, c.rope_dim);
        let tables = (&a.cos, &a.sin);
        let back = (&a.cos, &a.neg_sin);
        // y = x * w (+ bias): a bias goes through `tmp` because the matmul writes its output whole.
        let lin = |x: &GpuBuf,
                   w: &GpuBuf,
                   bias: &GpuBuf,
                   y: &GpuBuf,
                   tmp: &GpuBuf,
                   (k, n): (usize, usize)|
         -> Result<(), GpuError> {
            if bias.is_empty() {
                self.matmul.submit(g, x, w, y, (r, k, n))
            } else {
                self.matmul.submit(g, x, w, tmp, (r, k, n))?;
                ops::add_bias(g, tmp, bias, y)
            }
        };
        // Forward.
        gather_rows(g, &p[0], ids, &a.x[0], (r, d))?;
        for l in 0..nl {
            rmsnorm(g, &a.x[l], &p[pi(l, ATTN_NORM)], &a.h1[l], (r, d), eps)?;
            lin(
                &a.h1[l],
                &p[pi(l, WQ)],
                &p[pi(l, BQ)],
                &a.q[l],
                &a.tmp_q,
                (d, qd),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, WK)],
                &p[pi(l, BK)],
                &a.k[l],
                &a.tmp_kv,
                (d, kvd),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, WV)],
                &p[pi(l, BV)],
                &a.v[l],
                &a.tmp_kv,
                (d, kvd),
            )?;
            rope(g, &a.q[l], tables, (r, c.heads, hd, rd), ty, 0)?;
            rope(g, &a.k[l], tables, (r, c.kv_heads, hd, rd), ty, 0)?;
            let shape = (b, t, c.heads, c.kv_heads, hd);
            attention_gqa(
                g,
                (&a.q[l], &a.k[l], &a.v[l]),
                (&a.probs[l], &a.a[l]),
                shape,
            )?;
            self.matmul
                .submit(g, &a.a[l], &p[pi(l, WO)], &a.o, (r, qd, d))?;
            add(g, &a.x[l], &a.o, &a.x2[l])?;
            rmsnorm(g, &a.x2[l], &p[pi(l, FFN_NORM)], &a.h2[l], (r, d), eps)?;
            self.matmul
                .submit(g, &a.h2[l], &p[pi(l, GATE)], &a.gate[l], (r, d, h))?;
            self.matmul
                .submit(g, &a.h2[l], &p[pi(l, UP)], &a.up[l], (r, d, h))?;
            swiglu(g, &a.gate[l], &a.up[l], &a.act[l])?;
            self.matmul
                .submit(g, &a.act[l], &p[pi(l, DOWN)], &a.f2, (r, h, d))?;
            add(g, &a.x2[l], &a.f2, &a.x[l + 1])?;
        }
        let (on, head) = (pi(nl, 0), pi(nl, 1));
        rmsnorm(g, &a.x[nl], &p[on], &a.hf, (r, d), eps)?;
        self.matmul
            .submit(g, &a.hf, &p[head], &a.logits, (r, d, v))?;
        ops::cross_entropy(
            g,
            &a.logits,
            targets,
            (&a.dlogits, &a.row_loss, &a.loss),
            (r, v),
        )?;
        // Backward.
        lin_bwd(
            g,
            (&a.hf, &p[head], &a.dlogits),
            (&a.dhf, &gr[head]),
            (r, d, v),
        )?;
        rmsnorm_bwd(
            g,
            (&a.x[nl], &p[on], &a.dhf),
            (&a.dx_a, &gr[on]),
            &a.stats,
            (r, d),
            eps,
        )?;
        let (mut cur, mut next) = (&a.dx_a, &a.dx_b);
        for l in (0..nl).rev() {
            lin_bwd(
                g,
                (&a.act[l], &p[pi(l, DOWN)], cur),
                (&a.dact, &gr[pi(l, DOWN)]),
                (r, h, d),
            )?;
            swiglu_bwd(g, (&a.gate[l], &a.up[l], &a.dact), (&a.dgate, &a.dup))?;
            lin_bwd(
                g,
                (&a.h2[l], &p[pi(l, GATE)], &a.dgate),
                (&a.dh2a, &gr[pi(l, GATE)]),
                (r, d, h),
            )?;
            lin_bwd(
                g,
                (&a.h2[l], &p[pi(l, UP)], &a.dup),
                (&a.dh2b, &gr[pi(l, UP)]),
                (r, d, h),
            )?;
            add(g, &a.dh2a, &a.dh2b, &a.dh2)?;
            rmsnorm_bwd(
                g,
                (&a.x2[l], &p[pi(l, FFN_NORM)], &a.dh2),
                (&a.dx2n, &gr[pi(l, FFN_NORM)]),
                &a.stats,
                (r, d),
                eps,
            )?;
            add(g, cur, &a.dx2n, &a.dx2)?;
            lin_bwd(
                g,
                (&a.a[l], &p[pi(l, WO)], &a.dx2),
                (&a.da, &gr[pi(l, WO)]),
                (r, qd, d),
            )?;
            attention_gqa_bwd(
                g,
                (&a.q[l], &a.k[l], &a.v[l], &a.probs[l], &a.da),
                (&a.dq, &a.dk, &a.dv),
                (&a.dot, &a.ds),
                (b, t, c.heads, c.kv_heads, hd),
            )?;
            rope(g, &a.dq, back, (r, c.heads, hd, rd), ty, 0)?;
            rope(g, &a.dk, back, (r, c.kv_heads, hd, rd), ty, 0)?;
            for (wi, bi, dy, dh, n) in [
                (WQ, BQ, &a.dq, &a.dh1a, qd),
                (WK, BK, &a.dk, &a.dh1b, kvd),
                (WV, BV, &a.dv, &a.dh1c, kvd),
            ] {
                lin_bwd(
                    g,
                    (&a.h1[l], &p[pi(l, wi)], dy),
                    (dh, &gr[pi(l, wi)]),
                    (r, d, n),
                )?;
                if !p[pi(l, bi)].is_empty() {
                    colsum(g, dy, &gr[pi(l, bi)], (r, n))?;
                }
            }
            add(g, &a.dh1a, &a.dh1b, &a.t1)?;
            add(g, &a.t1, &a.dh1c, &a.dh1)?;
            rmsnorm_bwd(
                g,
                (&a.x[l], &p[pi(l, ATTN_NORM)], &a.dh1),
                (&a.dxn, &gr[pi(l, ATTN_NORM)]),
                &a.stats,
                (r, d),
                eps,
            )?;
            add(g, &a.dx2, &a.dxn, next)?;
            std::mem::swap(&mut cur, &mut next);
        }
        // Only the token-embedding gradient is wanted; the position kernel writes a scratch.
        embed_bwd(g, cur, ids_host, (&gr[0], &a.junk_pos), (v, d, t, c.ctx))
    }
}
