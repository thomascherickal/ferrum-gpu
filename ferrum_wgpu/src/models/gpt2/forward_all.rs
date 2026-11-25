use super::*;

impl<'g> GpuGpt2<'g> {
    /// Stateless full causal forward pass over `ids`; returns every position's logits
    /// (`ids.len() * vocab`). Uses batched kernels (matmul, not GEMV).
    pub fn forward_all(&self, ids: &[usize]) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        let t = ids.len();
        if t == 0 || t > c.ctx {
            return Err(err(format!("sequence length {t} must be in 1..={}", c.ctx)));
        }
        let (g, d) = (self.gpu, c.dim);
        let z = |n: usize| g.zeros(n);
        let gi = g.upload_indices(ids, c.vocab)?;
        let (x, x2, h) = (z(t * d)?, z(t * d)?, z(t * d)?);
        let (q, k, v, a, o) = (z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?);
        let (f1, act, f2) = (z(t * c.hidden)?, z(t * c.hidden)?, z(t * d)?);
        let (tmp_d, tmp_h) = (z(t * d)?, z(t * c.hidden)?);
        let probs = z(c.heads * t * t)?;
        let logits = z(t * c.vocab)?;
        g.batched(|| {
            ops::embed(g, (&self.tok, &self.pos_tab, &gi), &x, (t, d, t))?;
            // y = h * w + bias, all resident.
            let lin = |h: &GpuBuf,
                       w: &GpuBuf,
                       b: &GpuBuf,
                       y: &GpuBuf,
                       tmp: &GpuBuf,
                       (k, n): (usize, usize)| {
                self.matmul.submit(g, h, w, tmp, (t, k, n))?;
                ops::add_bias(g, tmp, b, y)
            };
            for b in &self.blocks {
                ops::layernorm(g, &x, &b.ln1_g, &b.ln1_b, &h, (t, d))?;
                lin(&h, &b.wq, &b.bq, &q, &tmp_d, (d, d))?;
                lin(&h, &b.wk, &b.bk, &k, &tmp_d, (d, d))?;
                lin(&h, &b.wv, &b.bv, &v, &tmp_d, (d, d))?;
                ops::attention(g, (&q, &k, &v), (&probs, &a), (1, t, c.heads, c.head_dim()))?;
                lin(&a, &b.wo, &b.bo, &o, &tmp_d, (d, d))?;
                ops::add(g, &x, &o, &x2)?;
                ops::layernorm(g, &x2, &b.ln2_g, &b.ln2_b, &h, (t, d))?;
                lin(&h, &b.w1, &b.b1, &f1, &tmp_h, (d, c.hidden))?;
                ops::gelu(g, &f1, &act)?;
                lin(&act, &b.w2, &b.b2, &f2, &tmp_d, (c.hidden, d))?;
                ops::add(g, &x2, &f2, &x)?;
            }
            ops::layernorm(g, &x, &self.lnf_g, &self.lnf_b, &h, (t, d))?;
            self.matmul
                .submit(g, &h, &self.head, &logits, (t, d, c.vocab))?;
            Ok(())
        })?;
        g.read(&logits)
    }

    /// Score `ids`: for each `i < len-1`, the log-probability of `ids[i+1]` given `ids[..=i]`
    /// and whether it was the (first) argmax. Only `2 * (len-1)` floats leave the device.
    pub fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, GpuError> {
        let c = self.cfg;
        let t = ids.len();
        if t < 2 {
            return Err(err("scoring needs at least 2 tokens"));
        }
        // The batched pass already ends in a logits buffer; run it again here resident so only
        // the score pairs are read back.
        let logits_host = self.forward_logits_buf(&ids[..t - 1])?;
        let rows = t - 1;
        let targets = self.gpu.upload_indices(&ids[1..], c.vocab)?;
        let out = self.gpu.zeros(2 * rows)?;
        ops::score(self.gpu, &logits_host, &targets, &out, (rows, c.vocab))?;
        let r = self.gpu.read(&out)?;
        Ok(r.chunks_exact(2).map(|p| (p[0], p[1] == 1.0)).collect())
    }
}
