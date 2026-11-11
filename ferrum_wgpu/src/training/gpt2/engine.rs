//! The shared [`Engine`] interface over the GPU trainer.

use super::{Engine, GpuTrainer, Snapshot, StepConfig, StepStats};

impl Engine for GpuTrainer<'_> {
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String> {
        GpuTrainer::step(self, ids, targets, batch, cfg).map_err(|e| e.to_string())
    }

    fn steps(&self) -> u64 {
        self.t
    }

    fn snapshot(&self) -> Result<Snapshot, String> {
        let (m, v) = self.moments_host().map_err(|e| e.to_string())?;
        Ok(Snapshot {
            weights: self.weights_host().map_err(|e| e.to_string())?,
            m,
            v,
            steps: self.t,
        })
    }

    fn restore(&mut self, s: &Snapshot) -> Result<(), String> {
        self.load(&s.weights, Some((&s.m, &s.v)), s.steps)
            .map_err(|e| e.to_string())
    }
}
