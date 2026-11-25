use super::*;

impl<'g> GpuTrainer<'g> {
    /// Upload `w`. Weights, gradients and moments are planned and checked against the adapter
    /// and `budget` before anything is allocated; activations are checked when a batch shape is
    /// first seen (including the static part, so the whole footprint is bounded).
    pub fn new(gpu: &'g Gpu, w: &Gpt2, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        static_plan(&w.cfg)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        Ok(GpuTrainer {
            gpu,
            cfg: w.cfg,
            budget,
            state: DeviceParams::upload(gpu, &w.params(), |i| w.decays(i))?,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            acts: None,
            accum: None,
            last_micro: 0,
            qat: false,
            tie: false,
            dropout: (0.0, 0),
            drop_offset: Cell::new(0),
            fwd: None,
            t: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Optimizer steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    pub(crate) fn host(&self, bufs: &[GpuBuf]) -> Result<Gpt2, GpuError> {
        let mut out = Gpt2::random(self.cfg, 1);
        for (dst, b) in out.params_mut().into_iter().zip(bufs) {
            *dst = self.gpu.read(b)?;
        }
        Ok(out)
    }

    /// Copy the weights back to the host.
    pub fn weights_host(&self) -> Result<Gpt2, GpuError> {
        self.host(&self.state.params)
    }

    /// Copy the gradients of the last forward/backward back to the host.
    pub fn grads_host(&self) -> Result<Gpt2, GpuError> {
        self.host(&self.state.grads)
    }
}
