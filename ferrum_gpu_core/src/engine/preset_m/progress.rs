use super::*;

/// Live state of a training run, shared with the HTTP thread.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    /// Steps finished.
    pub step: usize,
    /// Steps planned.
    pub total: usize,
    /// Mean loss of every finished step.
    pub losses: Vec<f32>,
    /// Set by the HTTP thread to end the run early.
    pub stop: bool,
    /// Loss on the held-out text before training (a run with a validation share only).
    pub val_before: Option<f32>,
    /// Loss on the held-out text after training.
    pub val_after: Option<f32>,
}

/// `batch` windows of `context` tokens and their next-token targets, starting at random offsets.
pub fn make_batch(
    text: &[u8],
    context: usize,
    batch: usize,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<usize>) {
    let starts = text.len() - context; // windows need one byte past the end for the target
    let (mut ids, mut targets) = (
        Vec::with_capacity(batch * context),
        Vec::with_capacity(batch * context),
    );
    for _ in 0..batch {
        let s = rng.below(starts);
        ids.extend(text[s..s + context].iter().map(|&b| usize::from(b)));
        targets.extend(text[s + 1..s + context + 1].iter().map(|&b| usize::from(b)));
    }
    (ids, targets)
}

/// One optimizer step: `(ids, targets, batch, config)` to the step's stats.
pub type StepFn<'a> =
    dyn FnMut(&[usize], &[usize], usize, &StepConfig) -> Result<StepStats, String> + 'a;

/// The progress record, recovered if a panicking thread poisoned the lock (it is plain counters).
pub(crate) fn lock(progress: &Arc<Mutex<Progress>>) -> std::sync::MutexGuard<'_, Progress> {
    progress
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
