use super::*;

impl<'g> GpuTrainer<'g> {
    /// Like [`step`](Self::step) but with an explicit micro-batch size, which must divide `batch`.
    pub fn step_with_micro(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        micro: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        if micro == 0 || !batch.is_multiple_of(micro) {
            return Err(err(format!(
                "micro-batch {micro} must be positive and divide the batch of {batch}"
            )));
        }
        self.check_fits(batch, micro, t)?;
        self.step_split(ids, targets, (batch, micro, t), cfg)
    }

    /// Weights and Adam moments together, for tests that compare whole training states.
    #[cfg(test)]
    pub(super) fn snapshot_pair(&self) -> (Gpt2, (Gpt2, Gpt2)) {
        (self.weights_host().unwrap(), self.moments_host().unwrap())
    }

    /// Sequences per micro-batch used by the last step (equals the batch when it was not split).
    pub fn last_micro(&self) -> usize {
        self.last_micro
    }

    /// The memory plan of running `batch` sequences as micro-batches of `micro`.
    pub(crate) fn footprint(&self, batch: usize, micro: usize, t: usize) -> BufferPlan {
        let mut plan = static_plan(&self.cfg);
        if micro < batch {
            accum_plan(&self.cfg)
                .rows()
                .iter()
                .for_each(|(n, b)| plan.add(n, *b));
        }
        if self.qat {
            qat_plan(&self.cfg)
                .rows()
                .iter()
                .for_each(|(n, b)| plan.add(n, *b));
        }
        activation_plan(&self.cfg, micro, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan
    }

    pub(super) fn check_fits(&self, batch: usize, micro: usize, t: usize) -> Result<(), GpuError> {
        self.footprint(batch, micro, t)
            .check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| {
                err(format!(
                    "batch {batch} x {t} tokens does not fit in micro-batches of {micro}: {e}"
                ))
            })
    }
}
