use super::*;

impl<'g> GpuLlama<'g> {
    /// Consume one token at the next position and return its next-token logits row.
    pub fn feed(&mut self, token: usize) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        if self.pos >= c.ctx {
            return Err(err(format!("context of {} tokens is full", c.ctx)));
        }
        let (g, s, d) = (self.gpu, &self.s, c.dim);
        let (qd, kvd) = (c.q_dim(), c.kv_dim());
        let id = g.upload_indices(&[token], c.vocab)?;
        // The whole token is recorded into one command encoder and submitted once.
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            let mm = &self.matmul;
            g.copy_f32(&id, 0, &s.ids, 0, 1)?;
            self.tok.gather(g, (&[token], &s.ids), &s.x, d)?;
            for b in &self.blocks {
                ops_llama::rmsnorm(g, &s.x, &b.attn_norm, &s.h, (1, d), c.eps)?;
                b.wq.apply(
                    g,
                    mm,
                    (&s.h, b.bq.as_ref(), self.zero(qd)),
                    (&s.q, &s.tmp_q),
                    1,
                )?;
                b.wk.apply(
                    g,
                    mm,
                    (&s.h, b.bk.as_ref(), self.zero(kvd)),
                    (&s.k, &s.tmp_kv),
                    1,
                )?;
                b.wv.apply(
                    g,
                    mm,
                    (&s.h, b.bv.as_ref(), self.zero(kvd)),
                    (&s.v, &s.tmp_kv),
                    1,
                )?;
                let tables = (&self.cos, &self.sin);
                let (hd, rd, ty) = (c.head_dim, c.rope_dim, c.rope_type);
                ops_llama::rope(g, &s.q, tables, (1, c.heads, hd, rd), ty, self.pos)?;
                ops_llama::rope(g, &s.k, tables, (1, c.kv_heads, hd, rd), ty, self.pos)?;
                g.copy_f32(&s.k, 0, &b.kcache, self.pos * kvd, kvd)?;
                g.copy_f32(&s.v, 0, &b.vcache, self.pos * kvd, kvd)?;
                let dims = (c.heads, c.kv_heads, hd, self.pos + 1, c.ctx);
                ops_llama::attention_decode_gqa(
                    g,
                    (&s.q, &b.kcache, &b.vcache),
                    (&s.probs, &s.attn),
                    dims,
                )?;
                b.wo.apply(g, mm, (&s.attn, None, self.zero(d)), (&s.o, &s.tmp_q), 1)?;
                ops::add(g, &s.x, &s.o, &s.x2)?;
                ops_llama::rmsnorm(g, &s.x2, &b.ffn_norm, &s.h, (1, d), c.eps)?;
                b.gate.apply(
                    g,
                    mm,
                    (&s.h, None, self.zero(c.hidden)),
                    (&s.gate, &s.tmp_q),
                    1,
                )?;
                b.up.apply(
                    g,
                    mm,
                    (&s.h, None, self.zero(c.hidden)),
                    (&s.up, &s.tmp_q),
                    1,
                )?;
                ops_llama::swiglu(g, &s.gate, &s.up, &s.act)?;
                b.down
                    .apply(g, mm, (&s.act, None, self.zero(d)), (&s.f2, &s.tmp_q), 1)?;
                ops::add(g, &s.x2, &s.f2, &s.x)?;
            }
            ops_llama::rmsnorm(g, &s.x, &self.out_norm, &s.h, (1, d), c.eps)?;
            self.head.apply(
                g,
                mm,
                (&s.h, None, self.zero(c.vocab)),
                (&s.logits, &s.tmp_q),
                1,
            )?;
            Ok(())
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        self.pos += 1;
        g.read(&s.logits)
    }
}
