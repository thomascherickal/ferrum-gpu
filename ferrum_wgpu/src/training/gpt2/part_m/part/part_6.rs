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
        for (i, g) in source.iter().enumerate().filter(|(i, _)| Some(*i) != skip) {
            sumsq_parts(self.gpu, g, &self.part, self.part_offsets[i])?;
        }
        norm_scale(self.gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
        for (i, g) in source.iter().enumerate().filter(|(i, _)| Some(*i) != skip) {
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

    /// Allocate the gradient accumulators (once, on first use of micro-batching).
    pub(crate) fn ensure_accum(&mut self) -> Result<(), GpuError> {
        if self.accum.is_none() {
            let bufs = self
                .params
                .iter()
                .map(|p| self.gpu.zeros(p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            self.accum = Some(bufs);
        }
        Ok(())
    }
}
