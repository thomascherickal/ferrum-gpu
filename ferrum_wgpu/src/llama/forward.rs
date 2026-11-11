//! Batched passes of [`GpuLlama`]: full causal forward over a sequence, and scoring.

use super::{err, GpuLlama};
use crate::device::GpuError;
use crate::ops;
use crate::ops_llama;
use crate::tensor::GpuBuf;

impl GpuLlama<'_> {
    /// Stateless full causal forward pass over `ids`; returns every position's logits
    /// (`ids.len() * vocab`). Uses batched kernels (matmul, not GEMV) and does not touch the
    /// KV cache.
    pub fn forward_all(&self, ids: &[usize]) -> Result<Vec<f32>, GpuError> {
        self.gpu.read(&self.forward_buf(ids)?)
    }

    /// The same pass, leaving the logits on the device.
    fn forward_buf(&self, ids: &[usize]) -> Result<GpuBuf, GpuError> {
        let c = self.cfg;
        let t = ids.len();
        if t == 0 || t > c.ctx {
            return Err(err(format!("sequence length {t} must be in 1..={}", c.ctx)));
        }
        let (g, d) = (self.gpu, c.dim);
        let (qd, kvd, hid) = (c.q_dim(), c.kv_dim(), c.hidden);
        let z = |n: usize| g.zeros(n);
        let gi = g.upload_indices(ids, c.vocab)?;
        let (x, x2, h, o, f2) = (z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?);
        let (q, a) = (z(t * qd)?, z(t * qd)?);
        let (k, v) = (z(t * kvd)?, z(t * kvd)?);
        let (gate, up, act) = (z(t * hid)?, z(t * hid)?, z(t * hid)?);
        let (tmp_q, tmp_kv) = (z(t * qd)?, z(t * kvd)?);
        let probs = z(c.heads * t * t)?;
        let logits = z(t * c.vocab)?;
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            let mm = &self.matmul;
            ops_llama::gather_rows(g, &self.tok, &gi, &x, (t, d))?;
            for b in &self.blocks {
                ops_llama::rmsnorm(g, &x, &b.attn_norm, &h, (t, d), c.eps)?;
                b.wq.apply(g, mm, (&h, b.bq.as_ref(), self.zero(qd)), (&q, &tmp_q), t)?;
                b.wk.apply(g, mm, (&h, b.bk.as_ref(), self.zero(kvd)), (&k, &tmp_kv), t)?;
                b.wv.apply(g, mm, (&h, b.bv.as_ref(), self.zero(kvd)), (&v, &tmp_kv), t)?;
                let tables = (&self.cos, &self.sin);
                let (hd, rd, ty) = (c.head_dim, c.rope_dim, c.rope_type);
                ops_llama::rope(g, &q, tables, (t, c.heads, hd, rd), ty, 0)?;
                ops_llama::rope(g, &k, tables, (t, c.kv_heads, hd, rd), ty, 0)?;
                ops_llama::attention_gqa(
                    g,
                    (&q, &k, &v),
                    (&probs, &a),
                    (1, t, c.heads, c.kv_heads, hd),
                )?;
                b.wo.apply(g, mm, (&a, None, self.zero(d)), (&o, &tmp_q), t)?;
                ops::add(g, &x, &o, &x2)?;
                ops_llama::rmsnorm(g, &x2, &b.ffn_norm, &h, (t, d), c.eps)?;
                b.gate
                    .apply(g, mm, (&h, None, self.zero(hid)), (&gate, &tmp_q), t)?;
                b.up.apply(g, mm, (&h, None, self.zero(hid)), (&up, &tmp_q), t)?;
                ops_llama::swiglu(g, &gate, &up, &act)?;
                b.down
                    .apply(g, mm, (&act, None, self.zero(d)), (&f2, &tmp_q), t)?;
                ops::add(g, &x2, &f2, &x)?;
            }
            ops_llama::rmsnorm(g, &x, &self.out_norm, &h, (t, d), c.eps)?;
            self.head
                .apply(g, mm, (&h, None, self.zero(c.vocab)), (&logits, &tmp_q), t)
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        Ok(logits)
    }

    /// Score `ids`: for each `i < len-1`, the log-probability of `ids[i+1]` given `ids[..=i]` and
    /// whether it was the (first) argmax. Only `2 * (len-1)` floats leave the device.
    pub fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, GpuError> {
        let t = ids.len();
        if t < 2 {
            return Err(err("scoring needs at least 2 tokens"));
        }
        let rows = t - 1;
        let logits = self.forward_buf(&ids[..rows])?;
        let targets = self.gpu.upload_indices(&ids[1..], self.cfg.vocab)?;
        let out = self.gpu.zeros(2 * rows)?;
        ops::score(self.gpu, &logits, &targets, &out, (rows, self.cfg.vocab))?;
        let r = self.gpu.read(&out)?;
        Ok(r.chunks_exact(2).map(|p| (p[0], p[1] == 1.0)).collect())
    }
}
