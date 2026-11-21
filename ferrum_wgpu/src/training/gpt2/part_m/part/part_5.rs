use super::*;

impl<'g> GpuTrainer<'g> {
    /// The largest micro-batch size (a divisor of `batch`) whose footprint fits. When nothing
    /// fits, the error is the one for a single sequence, the closest to fitting.
    pub(crate) fn choose_micro(&self, batch: usize, t: usize) -> Result<usize, GpuError> {
        let mut last = err("empty batch");
        for m in (1..=batch).rev().filter(|m| batch.is_multiple_of(*m)) {
            match self.check_fits(batch, m, t) {
                Ok(()) => return Ok(m),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    pub(crate) fn step_split(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        (batch, micro, t): (usize, usize, usize),
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let chunks = batch / micro;
        self.ensure_acts(micro, t)?;
        if chunks > 1 {
            self.ensure_accum()?;
        }
        let rows = micro * t;
        let (mut gid, mut gtg) = (Vec::with_capacity(chunks), Vec::with_capacity(chunks));
        for c in 0..chunks {
            let r = c * rows..(c + 1) * rows;
            gid.push(self.gpu.upload_indices(&ids[r.clone()], self.cfg.vocab)?);
            gtg.push(self.gpu.upload_indices(&targets[r], self.cfg.vocab)?);
        }
        let losses = self.gpu.zeros(chunks)?;
        let step = self.t + 1;
        self.gpu.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            let acts = self.acts.as_ref().expect("allocated");
            let weight = 1.0 / chunks as f32;
            if self.qat {
                self.record_snap()?; // once per optimizer step: every micro-batch sees the same weights
            }
            for c in 0..chunks {
                self.set_dropout_offset(c * rows * self.cfg.hidden)?;
                self.record(&ids[c * rows..(c + 1) * rows], &gid[c], &gtg[c])?;
                self.gpu.copy_f32(&acts.loss, 0, &losses, c, 1)?;
                if let Some(acc) = self.accum.as_ref().filter(|_| chunks > 1) {
                    for (a, g) in acc.iter().zip(&self.grads) {
                        accumulate(self.gpu, a, g, weight, c == 0)?;
                    }
                }
            }
            let source = match self.accum.as_ref() {
                Some(acc) if chunks > 1 => acc,
                _ => &self.grads,
            };
            if self.tie {
                self.record_fold_head_grad(source)?;
            }
            self.optimize(source, step, cfg)?;
            if self.tie {
                self.record_derive_head(&self.params)?;
            }
            Ok(())
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        self.t = step;
        self.last_micro = micro;
        let per_chunk = self.gpu.read(&losses)?;
        let loss = per_chunk.iter().sum::<f32>() / chunks as f32;
        let norm = self.gpu.read(&self.scale)?[1];
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }
}
