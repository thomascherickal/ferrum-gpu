use super::*;

/// What one optimizer step needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepConfig {
    /// AdamW hyperparameters (`lr` is this step's, already scheduled).
    pub adam: Adam,
    /// Global-norm clip cap; `0.0` disables clipping.
    pub clip: f32,
}

/// What a step reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// Mean cross-entropy of the batch (before the update).
    pub loss: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Everything an engine needs to continue a run: weights, Adam moments, steps taken.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Weights.
    pub weights: Gpt2,
    /// First moments.
    pub m: Gpt2,
    /// Second moments.
    pub v: Gpt2,
    /// Optimizer steps completed.
    pub steps: u64,
}

/// A backend that can run training steps.
pub trait Engine {
    /// One forward/backward/clip/AdamW step on `batch` equal-length sequences.
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String>;
    /// Optimizer steps completed.
    fn steps(&self) -> u64;
    /// Copy the full training state to the host.
    fn snapshot(&self) -> Result<Snapshot, String>;
    /// Replace the full training state.
    fn restore(&mut self, s: &Snapshot) -> Result<(), String>;
}

/// The CPU engine (also the oracle for the GPU engine).
pub struct CpuTrainer {
    pub(super) w: Gpt2,
    pub(super) state: AdamState,
    pub(super) threads: usize,
    pub(super) qat: bool,
    pub(super) tie: bool,
    /// FFN-hidden dropout rate and the run's base seed (rate 0 is off).
    pub(super) dropout: (f32, u64),
}
