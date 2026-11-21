use super::*;

impl<'g> GpuGpt2<'g> {
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
}
