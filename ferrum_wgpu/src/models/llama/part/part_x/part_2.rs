use super::*;

impl<'g> GpuLlama<'g> {
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
}
