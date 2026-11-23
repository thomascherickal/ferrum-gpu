use super::*;

impl<'g> GpuTrainer<'g> {
    /// Copy the Adam moments back to the host as `(m, v)`.
    pub fn moments_host(&self) -> Result<(Gpt2, Gpt2), GpuError> {
        Ok((self.host(&self.m)?, self.host(&self.v)?))
    }

    /// Replace weights, Adam moments and the step counter (resume from a checkpoint).
    pub fn load(
        &mut self,
        w: &Gpt2,
        moments: Option<(&Gpt2, &Gpt2)>,
        steps: u64,
    ) -> Result<(), GpuError> {
        if w.cfg != self.cfg {
            return Err(err("load: the model shape differs from the trainer's"));
        }
        let (zm, zv) = (w.zeros_like(), w.zeros_like());
        let (m, v) = moments.unwrap_or((&zm, &zv));
        if m.cfg != self.cfg || v.cfg != self.cfg {
            return Err(err("load: the moment shapes differ from the trainer's"));
        }
        for (set, src) in [(&self.params, w), (&self.m, m), (&self.v, v)] {
            for (b, p) in set.iter().zip(src.params()) {
                self.gpu.copy_from_host(b, p)?;
            }
        }
        self.t = steps;
        if self.tie {
            self.derive_head_now()?; // a snapshot from an untied run is tied on the way in
        }
        Ok(())
    }

    pub(crate) fn validate(
        &self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<usize, GpuError> {
        crate::training::batch_check::check_batch(ids, targets, batch, self.cfg.ctx, self.cfg.vocab)
    }
}
