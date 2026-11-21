use super::*;

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            context: 16,
            batch: 16,
            stride: 0,
            epochs: 1,
            max_steps: 0,
            lr: 0.01,
            weight_decay: 0.0,
            clip: 0.0,
            warmup: 0,
            cosine: false,
            seed: 1337,
            checkpoint_every: 0,
        }
    }
}

/// The run parameters a checkpoint must be resumed with (changing any changes the window order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunShape {
    /// Window length.
    pub context: u64,
    /// Windows per step.
    pub batch: u64,
    /// Window stride (resolved, never 0).
    pub stride: u64,
    /// Windows in the corpus.
    pub windows: u64,
    /// Order seed.
    pub seed: u64,
}

/// A resumable training state: the engine snapshot plus where the loop was.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// Engine state.
    pub snapshot: Snapshot,
    /// Epoch in progress (0-based).
    pub epoch: u64,
    /// Steps already done within that epoch.
    pub step_in_epoch: u64,
    /// The shape of the run that produced it.
    pub shape: RunShape,
    /// Identity of the corpus, if the caller recorded one.
    pub corpus_id: Option<String>,
}

/// Callbacks for a run. All have no-op defaults.
pub trait Hooks {
    /// After every step; return `false` to stop (a final checkpoint is still offered).
    fn on_step(&mut self, _step: u64, _stats: &StepStats) -> bool {
        true
    }
    /// A checkpoint is due (cadence, stop, or end of run).
    fn on_checkpoint(&mut self, _cp: &Checkpoint) -> Result<(), String> {
        Ok(())
    }
}

/// Hooks that do nothing.
pub struct NoHooks;
impl Hooks for NoHooks {}
