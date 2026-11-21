use super::*;

impl<'g> GpuLlamaTrainer<'g> {
    pub(crate) fn validate(
        &self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<usize, GpuError> {
        let c = &self.cfg;
        if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
            return Err(err(format!(
                "{} tokens do not split into {batch} sequences",
                ids.len()
            )));
        }
        let t = ids.len() / batch;
        if t > c.ctx {
            return Err(err(format!(
                "sequence length {t} exceeds the context {}",
                c.ctx
            )));
        }
        if targets.len() != ids.len() {
            return Err(err(format!(
                "{} targets for {} tokens",
                targets.len(),
                ids.len()
            )));
        }
        if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= c.vocab) {
            return Err(err(format!(
                "token {bad} is out of range for vocab {}",
                c.vocab
            )));
        }
        Ok(t)
    }

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
        let mut plan = static_plan(&self.template_with_sizes());
        activation_plan(&self.cfg, batch, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan.check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| err(format!("batch {batch} x {t} tokens does not fit: {e}")))?;
        let rope = self.template.rope.clone();
        self.acts = Some(Acts::new(self.gpu, &self.cfg, &rope, batch, t)?);
        Ok(())
    }

    /// The template with the parameter sizes filled in as zero vectors (for the static plan).
    pub(crate) fn template_with_sizes(&self) -> Llama {
        let mut m = self.template.clone();
        for (dst, b) in m.params_mut().into_iter().zip(&self.params) {
            *dst = vec![0.0; b.len()];
        }
        m
    }
}
