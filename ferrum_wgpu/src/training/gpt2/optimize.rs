use super::*;

impl<'g> GpuTrainer<'g> {
    /// Norm, clip scale and AdamW over the gradients in `source` (the recorded commands only).
    pub(crate) fn optimize(
        &self,
        source: &[GpuBuf],
        step: u64,
        cfg: &StepConfig,
    ) -> Result<(), GpuError> {
        // a tied head has no gradient of its own (it was folded into the embedding's)
        let skip = self.tie.then(|| self.head_index());
        let active = |i, _: &GpuBuf| Some(i) != skip;
        self.state.optimize(self.gpu, source, active, step, cfg)
    }

    /// Allocate the gradient accumulators (once, on first use of micro-batching).
    pub(super) fn ensure_accum(&mut self) -> Result<(), GpuError> {
        if self.accum.is_none() {
            let bufs = self
                .state
                .params
                .iter()
                .map(|p| self.gpu.zeros(p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            self.accum = Some(bufs);
        }
        Ok(())
    }
}
