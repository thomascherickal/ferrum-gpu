use super::*;

impl GpuTrainer<'_> {
    /// Record the snap of every parameter into the forward-weights buffers (a batch must be open).
    pub(crate) fn record_snap(&self) -> Result<(), GpuError> {
        let fwd = self.fwd.as_ref().expect("qat buffers allocated");
        for (i, (src, dst)) in self.state.params.iter().zip(fwd).enumerate() {
            match snap_kind(&self.cfg, i) {
                Snap::Rows(ne0) => q8_snap_rows(self.gpu, src, dst, ne0)?,
                Snap::Cols(r, c) => q8_snap_cols(self.gpu, src, dst, (r, c))?,
                Snap::Keep => self.gpu.copy_f32(src, 0, dst, 0, src.len())?,
            }
        }
        if self.tie {
            self.record_derive_head(fwd)?; // the head the forward pass sees is the snapped embedding's
        }
        Ok(())
    }

    /// The weights the forward and backward pass read: the snapped copies under QAT, else the masters.
    pub(crate) fn forward_weights(&self) -> &Vec<GpuBuf> {
        match self.fwd.as_ref() {
            Some(f) if self.qat => f,
            _ => &self.state.params,
        }
    }

    /// Turn quantisation-aware training on or off. Enabling allocates one more copy of the
    /// parameters (checked against the budget first); disabling frees it.
    pub fn set_qat(&mut self, on: bool) -> Result<(), GpuError> {
        if !on {
            self.qat = false;
            self.fwd = None;
            return Ok(());
        }
        if self.fwd.is_none() {
            let mut plan = crate::training::gpt2::plan::static_plan(&self.cfg);
            qat_plan(&self.cfg)
                .rows()
                .iter()
                .for_each(|(n, b)| plan.add(n, *b));
            plan.check(self.gpu.info().max_binding, self.budget)
                .map_err(|e| GpuError(format!("quantisation-aware training does not fit: {e}")))?;
            let bufs = self
                .state
                .params
                .iter()
                .map(|p| self.gpu.zeros(p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            self.fwd = Some(bufs);
        }
        self.qat = true;
        Ok(())
    }

    /// Whether quantisation-aware training is on.
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }
}
