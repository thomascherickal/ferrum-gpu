use super::*;

impl<'g> GpuLlamaTrainer<'g> {
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
        self.gpu.begin_batch()?;
        let rec = self.record(ids, &gid, &gtg);
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        Ok(self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0])
    }

    /// One full optimizer step: forward, backward, clip, AdamW. The loss and the pre-clip
    /// gradient norm are the only things read back.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        self.ensure_acts(batch, t)?;
        let (gid, gtg) = (
            self.gpu.upload_indices(ids, self.cfg.vocab)?,
            self.gpu.upload_indices(targets, self.cfg.vocab)?,
        );
        let step = self.t + 1;
        self.gpu.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            self.record(ids, &gid, &gtg)?;
            self.optimize(step, cfg)
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        self.t = step;
        let loss = self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0];
        let grad_norm = self.gpu.read(&self.scale)?[1];
        Ok(StepStats { loss, grad_norm })
    }

    /// Norm, clip scale and AdamW over the gradients (the recorded commands only). Absent
    /// biases are empty tensors and are skipped.
    pub(crate) fn optimize(&self, step: u64, cfg: &StepConfig) -> Result<(), GpuError> {
        for (i, g) in self.grads.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
            sumsq_parts(self.gpu, g, &self.part, self.part_offsets[i])?;
        }
        norm_scale(self.gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
        for (i, g) in self.grads.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
            adam_update(
                self.gpu,
                (&self.params[i], g, &self.m[i], &self.v[i]),
                &self.scale,
                &cfg.adam,
                self.decays[i],
                step,
            )?;
        }
        Ok(())
    }
}
