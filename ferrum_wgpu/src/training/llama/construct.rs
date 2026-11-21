use super::*;

impl<'g> GpuLlamaTrainer<'g> {
    /// Upload `w`. Weights, gradients and moments are planned and checked against the adapter
    /// and `budget` before anything is allocated; activations are checked (together with the
    /// static part) when a batch shape is first seen. Tied heads are refused.
    pub fn new(gpu: &'g Gpu, w: &Llama, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        w.rope.check(w.cfg.rope_dim).map_err(err)?;
        if w.tied {
            return Err(err(
                "training a tied head is not supported: train an untied copy",
            ));
        }
        static_plan(w)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let up = |m: &Llama| {
            m.params()
                .iter()
                .map(|p| gpu.upload(p))
                .collect::<Result<Vec<_>, _>>()
        };
        let zeros = w.zeros_like();
        let sizes: Vec<usize> = w.params().iter().map(|p| p.len()).collect();
        let mut part_offsets = Vec::with_capacity(sizes.len());
        let mut total = 0;
        for &n in &sizes {
            part_offsets.push(total);
            total += parts_for(n);
        }
        let mut template = w.clone();
        template
            .params_mut()
            .into_iter()
            .for_each(|p| *p = Vec::new());
        Ok(GpuLlamaTrainer {
            gpu,
            cfg: w.cfg,
            template,
            budget,
            params: up(w)?,
            grads: up(&zeros)?,
            m: up(&zeros)?,
            v: up(&zeros)?,
            decays: (0..sizes.len()).map(|i| w.decays(i)).collect(),
            part: gpu.zeros(total)?,
            part_offsets,
            scale: gpu.zeros(2)?,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            acts: None,
            t: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &ferrum_ref::llama::Config {
        &self.cfg
    }

    /// Optimizer steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    pub(crate) fn host(&self, bufs: &[GpuBuf]) -> Result<Llama, GpuError> {
        let mut out = self.template.clone();
        for (dst, b) in out.params_mut().into_iter().zip(bufs) {
            *dst = if b.is_empty() {
                Vec::new()
            } else {
                self.gpu.read(b)?
            };
        }
        Ok(out)
    }
}
