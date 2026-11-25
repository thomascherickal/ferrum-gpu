use super::*;

impl<'g> GpuLlamaTrainer<'g> {
    /// Check the batch, make its activations and upload its token ids and targets.
    fn prepare(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<(GpuBuf, GpuBuf), GpuError> {
        let t = self.validate(ids, targets, batch)?;
        self.ensure_acts(batch, t)?;
        let vocab = self.cfg.vocab;
        Ok((
            self.gpu.upload_indices(ids, vocab)?,
            self.gpu.upload_indices(targets, vocab)?,
        ))
    }

    /// The mean loss of the last recorded batch.
    fn read_loss(&self) -> Result<f32, GpuError> {
        Ok(self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").head.loss)?[0])
    }

    /// Forward and backward for one batch; leaves gradients in the device buffers and returns the
    /// mean loss. `ids[i]` is followed by `targets[i]`; `batch` sequences of equal length.
    pub fn loss_and_grads(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<f32, GpuError> {
        let (gid, gtg) = self.prepare(ids, targets, batch)?;
        self.gpu.batched(|| self.record(ids, &gid, &gtg))?;
        self.read_loss()
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
        let (gid, gtg) = self.prepare(ids, targets, batch)?;
        let step = self.t + 1;
        self.gpu.batched(|| {
            self.record(ids, &gid, &gtg)?;
            self.optimize(step, cfg)
        })?;
        self.t = step;
        let loss = self.read_loss()?;
        let grad_norm = self.gpu.read(&self.state.scale)?[1];
        Ok(StepStats { loss, grad_norm })
    }

    /// Norm, clip scale and AdamW over the gradients (the recorded commands only). Absent
    /// biases are empty tensors and are skipped.
    pub(crate) fn optimize(&self, step: u64, cfg: &StepConfig) -> Result<(), GpuError> {
        let present = |_, g: &GpuBuf| !g.is_empty();
        let s = &self.state;
        s.optimize(self.gpu, &s.grads, present, step, cfg)
    }
}
