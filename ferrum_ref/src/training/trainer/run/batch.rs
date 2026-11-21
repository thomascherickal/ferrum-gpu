use super::*;

/// The inputs and next-token targets of the windows `picked` (indices into `starts`).
pub(super) fn load_batch<T: TokenSource + ?Sized>(
    tokens: &T,
    starts: &[usize],
    picked: &[usize],
    context: usize,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut ids = Vec::with_capacity(picked.len() * context);
    let mut targets = Vec::with_capacity(picked.len() * context);
    for &w in picked {
        let win = tokens.window(starts[w], context + 1)?;
        ids.extend_from_slice(&win[..context]);
        targets.extend_from_slice(&win[1..]);
    }
    Ok((ids, targets))
}

/// AdamW at learning rate `lr` with the run's weight decay and clip.
pub(super) fn step_config(cfg: &TrainConfig, lr: f32) -> StepConfig {
    let adam = Adam {
        weight_decay: cfg.weight_decay,
        ..Adam::new(lr)
    };
    StepConfig {
        adam,
        clip: cfg.clip,
    }
}
