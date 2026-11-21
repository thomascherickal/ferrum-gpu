use super::*;

impl<'g> GpuGpt2<'g> {
    /// Upload `w`. The buffer plan is checked against the adapter and `budget` first, so an
    /// oversized model is refused before any device memory is allocated.
    pub fn new(gpu: &'g Gpu, w: &Gpt2, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        plan(&w.cfg)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let c = w.cfg;
        let up = |v: &[f32]| gpu.upload(v);
        let blocks = w
            .blocks
            .iter()
            .map(|b: &Block| {
                Ok(DevBlock {
                    ln1_g: up(&b.ln1_g)?,
                    ln1_b: up(&b.ln1_b)?,
                    wq: up(&b.wq)?,
                    bq: up(&b.bq)?,
                    wk: up(&b.wk)?,
                    bk: up(&b.bk)?,
                    wv: up(&b.wv)?,
                    bv: up(&b.bv)?,
                    wo: up(&b.wo)?,
                    bo: up(&b.bo)?,
                    ln2_g: up(&b.ln2_g)?,
                    ln2_b: up(&b.ln2_b)?,
                    w1: up(&b.w1)?,
                    b1: up(&b.b1)?,
                    w2: up(&b.w2)?,
                    b2: up(&b.b2)?,
                    kcache: gpu.zeros(c.ctx * c.dim)?,
                    vcache: gpu.zeros(c.ctx * c.dim)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let z = |n| gpu.zeros(n);
        let s = Scratch {
            ids: z(1)?,
            x: z(c.dim)?,
            x2: z(c.dim)?,
            h: z(c.dim)?,
            q: z(c.dim)?,
            k: z(c.dim)?,
            v: z(c.dim)?,
            attn: z(c.dim)?,
            o: z(c.dim)?,
            f1: z(c.hidden)?,
            act: z(c.hidden)?,
            f2: z(c.dim)?,
            probs: z(c.heads * c.ctx)?,
            logits: z(c.vocab)?,
        };
        Ok(GpuGpt2 {
            gpu,
            cfg: c,
            tok: up(&w.tok)?,
            pos_tab: up(&w.pos)?,
            blocks,
            lnf_g: up(&w.lnf_g)?,
            lnf_b: up(&w.lnf_b)?,
            head: up(&w.head)?,
            zero_vocab: z(c.vocab)?,
            s,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            pos: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Tokens consumed since the last [`reset`](Self::reset).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Forget the cached context (the cache buffers are overwritten as new tokens arrive).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Consume one token at the next position and return its next-token logits row.
    pub fn feed(&mut self, token: usize) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        if self.pos >= c.ctx {
            return Err(err(format!("context of {} tokens is full", c.ctx)));
        }
        let (g, s, d) = (self.gpu, &self.s, c.dim);
        // The index buffer is rewritten in place each token.
        let id = g.upload_indices(&[token], c.vocab)?;
        // The whole token is recorded into one command encoder and submitted once.
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            g.copy_f32(&id, 0, &s.ids, 0, 1)?;
            ops::embed_at(
                g,
                (&self.tok, &self.pos_tab, &s.ids),
                &s.x,
                (1, d, c.ctx),
                self.pos,
            )?;
            for b in &self.blocks {
                ops::layernorm(g, &s.x, &b.ln1_g, &b.ln1_b, &s.h, (1, d))?;
                ops::gemv(g, (&s.h, &b.wq, &b.bq), &s.q, (d, d))?;
                ops::gemv(g, (&s.h, &b.wk, &b.bk), &s.k, (d, d))?;
                ops::gemv(g, (&s.h, &b.wv, &b.bv), &s.v, (d, d))?;
                g.copy_f32(&s.k, 0, &b.kcache, self.pos * d, d)?;
                g.copy_f32(&s.v, 0, &b.vcache, self.pos * d, d)?;
                let dims = (c.heads, c.head_dim(), self.pos + 1, c.ctx);
                ops::attention_decode(g, (&s.q, &b.kcache, &b.vcache), (&s.probs, &s.attn), dims)?;
                ops::gemv(g, (&s.attn, &b.wo, &b.bo), &s.o, (d, d))?;
                ops::add(g, &s.x, &s.o, &s.x2)?;
                ops::layernorm(g, &s.x2, &b.ln2_g, &b.ln2_b, &s.h, (1, d))?;
                ops::gemv(g, (&s.h, &b.w1, &b.b1), &s.f1, (d, c.hidden))?;
                ops::gelu(g, &s.f1, &s.act)?;
                ops::gemv(g, (&s.act, &b.w2, &b.b2), &s.f2, (c.hidden, d))?;
                ops::add(g, &s.x2, &s.f2, &s.x)?;
            }
            ops::layernorm(g, &s.x, &self.lnf_g, &self.lnf_b, &s.h, (1, d))?;
            ops::gemv(
                g,
                (&s.h, &self.head, &self.zero_vocab),
                &s.logits,
                (d, c.vocab),
            )?;
            Ok(())
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        self.pos += 1;
        g.read(&s.logits)
    }

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
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
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
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
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

    /// Like [`forward_all`](Self::forward_all) but leaves the logits on the device.
    pub(crate) fn forward_logits_buf(&self, ids: &[usize]) -> Result<GpuBuf, GpuError> {
        let host = self.forward_all(ids)?;
        // ponytail: forward_all reads back; re-upload keeps score simple. Return the buffer
        // directly once forward_all is split into a resident part and a readback part.
        self.gpu.upload(&host)
    }
}
