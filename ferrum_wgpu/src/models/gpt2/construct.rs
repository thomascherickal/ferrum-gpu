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
        let shape = crate::models::ScratchShape {
            d: c.dim,
            qd: c.dim,
            kvd: c.dim,
            h: c.hidden,
            probs: c.heads * c.ctx,
            vocab: c.vocab,
        };
        let s = crate::models::Scratch::new(gpu, shape, z(c.hidden)?)?;
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
}
