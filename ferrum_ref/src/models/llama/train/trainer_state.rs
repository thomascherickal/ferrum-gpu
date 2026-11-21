use super::*;

impl LlamaTrainer {
    /// A trainer for `w` with `threads` matmul threads (the result does not depend on it).
    pub fn new(w: Llama, threads: usize) -> Self {
        let state = Moments::new(&w);
        LlamaTrainer { w, state, threads }
    }

    /// The current (master) weights.
    pub fn weights(&self) -> &Llama {
        &self.w
    }

    /// Optimizer steps completed.
    pub fn steps(&self) -> u64 {
        self.state.t
    }

    /// One forward/backward/clip/AdamW step on `batch` equal-length sequences.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String> {
        let (loss, g) = loss_and_grads(&self.w, ids, targets, batch, self.threads)?;
        let norm = global_norm(&g);
        adam_step(
            &mut self.w,
            &g,
            &mut self.state,
            &cfg.adam,
            clip_scale(norm, cfg.clip),
        );
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }
}
