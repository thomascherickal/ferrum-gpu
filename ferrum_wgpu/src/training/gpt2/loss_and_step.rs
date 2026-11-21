use super::*;

impl<'g> GpuTrainer<'g> {
    /// Make sure activations for `(batch, t)` exist, checking the whole footprint first.
    pub(crate) fn ensure_acts(&mut self, batch: usize, t: usize) -> Result<(), GpuError> {
        if self
            .acts
            .as_ref()
            .is_some_and(|a| a.batch == batch && a.t == t)
        {
            return Ok(());
        }
        self.acts = None; // free the old shape before allocating the new one
        let mut plan = static_plan(&self.cfg);
        activation_plan(&self.cfg, batch, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan.check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| err(format!("batch {batch} x {t} tokens does not fit: {e}")))?;
        self.acts = Some(Acts::new(self.gpu, &self.cfg, batch, t)?);
        Ok(())
    }

    /// Forward and backward for one batch; leaves gradients in the device buffers and returns the
    /// mean loss. `ids[i]` is followed by `targets[i]`; `batch` sequences of equal length.
    pub fn loss_and_grads(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<f32, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        self.ensure_acts(batch, t)?;
        let (gid, gtg) = (
            self.gpu.upload_indices(ids, self.cfg.vocab)?,
            self.gpu.upload_indices(targets, self.cfg.vocab)?,
        );
        self.set_dropout_offset(0)?;
        self.gpu.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            if self.qat {
                self.record_snap()?;
            }
            self.record(ids, &gid, &gtg)
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        Ok(self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0])
    }

    /// One full optimizer step: forward, backward, clip, AdamW. The loss and the pre-clip
    /// gradient norm are the only things read back.
    ///
    /// If the whole batch does not fit in the memory budget it is split into equal micro-batches
    /// (the largest divisor of `batch` that fits), whose gradients are accumulated on the device
    /// before a single optimizer step. The result matches the unsplit step up to floating-point
    /// summation order. Only when even one sequence at a time does not fit is the step refused.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        let micro = self.choose_micro(batch, t)?;
        self.step_split(ids, targets, (batch, micro, t), cfg)
    }
}
