use super::*;

/// Where training tokens come from: memory, or a file read window by window.
pub trait TokenSource {
    /// Number of tokens.
    fn len(&self) -> usize;
    /// True when there are no tokens.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// `len` tokens starting at `start` (the range is inside the source).
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String>;
}

impl TokenSource for [usize] {
    fn len(&self) -> usize {
        <[usize]>::len(self)
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        self.get(start..start + len)
            .map(<[usize]>::to_vec)
            .ok_or_else(|| format!("window {start}+{len} is outside {} tokens", self.len()))
    }
}

impl TokenSource for Vec<usize> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        self.as_slice().window(start, len)
    }
}

/// The first `len` tokens of another source (the rest is held out, for validation).
pub struct Prefix<'a, T: TokenSource + ?Sized> {
    /// The whole source.
    pub inner: &'a T,
    /// How many of its tokens are visible.
    pub len: usize,
}

impl<T: TokenSource + ?Sized> TokenSource for Prefix<'_, T> {
    fn len(&self) -> usize {
        self.len.min(self.inner.len())
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        if start + len > self.len() {
            return Err(format!(
                "window {start}+{len} is outside the first {} tokens",
                self.len()
            ));
        }
        self.inner.window(start, len)
    }
}

/// How a run walks the corpus and steps the optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// Tokens per training window (the model sees `context` inputs and predicts the next token of each).
    pub context: usize,
    /// Windows per step (the last step of an epoch may hold fewer).
    pub batch: usize,
    /// Tokens between window starts; `0` means `context` (non-overlapping).
    pub stride: usize,
    /// Epochs to run (ignored when `max_steps > 0`).
    pub epochs: usize,
    /// Total optimizer steps for the run; `0` means `epochs` full epochs.
    pub max_steps: u64,
    /// Peak learning rate.
    pub lr: f32,
    /// AdamW weight decay.
    pub weight_decay: f32,
    /// Global-norm clip cap (`0.0` = off).
    pub clip: f32,
    /// Linear warmup steps (only with `cosine`).
    pub warmup: u64,
    /// Warmup then cosine decay to zero over the whole run; otherwise a fixed `lr`.
    pub cosine: bool,
    /// Seed of the window order.
    pub seed: u64,
    /// Offer a checkpoint every this many steps (`0` = only at the end).
    pub checkpoint_every: u64,
}

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
