use super::*;

impl<'g> GpuLlama<'g> {
    /// Upload `w`, quantising eligible matrices to `quant` when given. The buffer plan is checked
    /// against the adapter and `budget` first, so an oversized model is refused before any device
    /// memory is allocated.
    pub fn new(
        gpu: &'g Gpu,
        w: &Llama,
        quant: Option<QKind>,
        budget: Option<Budget>,
    ) -> Result<Self, GpuError> {
        Self::with_binding_limit(gpu, w, quant, budget, gpu.info().max_binding)
    }

    /// Like [`new`](Self::new), treating `limit` bytes as the largest buffer binding: a weight or
    /// the embedding larger than that is stored as blocks (see `shard`). `limit` above the
    /// adapter's own would fail at allocation; below it, it lets a test exercise sharding on a
    /// small model.
    pub fn with_binding_limit(
        gpu: &'g Gpu,
        w: &Llama,
        quant: Option<QKind>,
        budget: Option<Budget>,
        limit: u64,
    ) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        w.rope.check(w.cfg.rope_dim).map_err(err)?;
        plan_limited(&w.cfg, quant, limit)
            .check(limit.min(gpu.info().max_binding), budget)
            .map_err(|e| err(e.to_string()))?;
        let c = w.cfg;
        let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
        let up = |v: &[f32]| gpu.upload(v);
        let bias = |v: &[f32]| (!v.is_empty()).then(|| gpu.upload(v)).transpose();
        let lin = |m: &[f32], k, n| Lin::upload(gpu, m, (k, n), quant, limit);
        let blocks = w
            .blocks
            .iter()
            .map(|b| {
                Ok(DevBlock {
                    attn_norm: up(&b.attn_norm)?,
                    wq: lin(&b.wq, d, qd)?,
                    bq: bias(&b.bq)?,
                    wk: lin(&b.wk, d, kvd)?,
                    bk: bias(&b.bk)?,
                    wv: lin(&b.wv, d, kvd)?,
                    bv: bias(&b.bv)?,
                    wo: lin(&b.wo, qd, d)?,
                    ffn_norm: up(&b.ffn_norm)?,
                    gate: lin(&b.gate, d, h)?,
                    up: lin(&b.up, d, h)?,
                    down: lin(&b.down, h, d)?,
                    kcache: gpu.zeros(c.ctx * kvd)?,
                    vcache: gpu.zeros(c.ctx * kvd)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let z = |n| gpu.zeros(n);
        let s = Scratch {
            ids: z(1)?,
            x: z(d)?,
            x2: z(d)?,
            h: z(d)?,
            q: z(qd)?,
            k: z(kvd)?,
            v: z(kvd)?,
            attn: z(qd)?,
            o: z(d)?,
            gate: z(h)?,
            up: z(h)?,
            act: z(h)?,
            f2: z(d)?,
            probs: z(c.heads * c.ctx)?,
            logits: z(c.vocab)?,
            tmp_q: z(qd)?,
            tmp_kv: z(kvd)?,
        };
        let mut zeros: Vec<(usize, GpuBuf)> = Vec::new();
        for n in [qd, kvd, d, h, c.vocab] {
            if !zeros.iter().any(|(m, _)| *m == n) {
                zeros.push((n, z(n)?));
            }
        }
        let (cos, sin) = rope_tables(0, c.ctx, c.rope_dim, c.rope_base, &w.rope);
        Ok(GpuLlama {
            gpu,
            cfg: c,
            quant,
            limit,
            tok: Embedding::upload(gpu, &w.tok, d, limit)?,
            cos: up(&cos)?,
            sin: up(&sin)?,
            blocks,
            out_norm: up(&w.out_norm)?,
            head: lin(&w.head, d, c.vocab)?,
            zeros,
            s,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            pos: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The largest buffer binding, in bytes, this model was built for: the adapter's own limit
    /// unless [`with_binding_limit`](Self::with_binding_limit) lowered it.
    pub fn binding_limit(&self) -> u64 {
        self.limit
    }

    /// The quantisation requested at upload (ineligible matrices stay f32).
    pub fn quant(&self) -> Option<QKind> {
        self.quant
    }

    /// Tokens consumed since the last [`reset`](Self::reset).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Forget the cached context (the cache buffers are overwritten as new tokens arrive).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub(crate) fn zero(&self, n: usize) -> &GpuBuf {
        &self
            .zeros
            .iter()
            .find(|(m, _)| *m == n)
            .expect("a zero bias exists for every projection width")
            .1
    }

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
