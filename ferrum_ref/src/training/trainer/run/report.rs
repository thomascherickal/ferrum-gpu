use super::*;

/// What a run reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Steps run by this call.
    pub steps_run: u64,
    /// Loss of every step this call ran.
    pub losses: Vec<f32>,
    /// Engine step counter at the end.
    pub final_step: u64,
    /// Stopped early by the hook (rather than finishing the budget).
    pub stopped: bool,
}

/// Window start offsets: every `stride` tokens while a full `context + 1` tokens remain.
pub fn window_starts(n_tokens: usize, context: usize, stride: usize) -> Vec<usize> {
    let stride = if stride == 0 { context } else { stride };
    if context == 0 || stride == 0 || n_tokens < context + 1 {
        return Vec::new();
    }
    (0..=n_tokens - context - 1).step_by(stride).collect()
}

/// The visiting order of `n` windows in epoch `epoch`: a seeded shuffle that depends only on
/// `(seed, epoch)`, so a resumed run reproduces it without any saved generator state.
pub fn epoch_order(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    Rng::new(seed ^ epoch.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15)).shuffle(&mut order);
    order
}

/// Steps in one epoch of `n` windows at `batch` windows per step.
pub fn steps_per_epoch(n: usize, batch: usize) -> u64 {
    n.div_ceil(batch.max(1)) as u64
}
