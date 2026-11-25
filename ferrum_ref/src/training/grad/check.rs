use super::*;

pub(crate) fn check(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
) -> Result<usize, String> {
    check_batch(ids, targets, batch, m.cfg.ctx, m.cfg.vocab)
}

/// Check that `ids` split into `batch` sequences of at most `ctx` tokens, that `targets` match
/// them and that every token is below `vocab`; returns the sequence length.
pub(crate) fn check_batch(
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    ctx: usize,
    vocab: usize,
) -> Result<usize, String> {
    if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
        return Err(format!(
            "{} tokens do not split into {batch} sequences",
            ids.len()
        ));
    }
    let t = ids.len() / batch;
    if t > ctx {
        return Err(format!("sequence length {t} exceeds the context {}", ctx));
    }
    if targets.len() != ids.len() {
        return Err(format!(
            "{} targets for {} tokens",
            targets.len(),
            ids.len()
        ));
    }
    if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= vocab) {
        return Err(format!("token {bad} is out of range for vocab {}", vocab));
    }
    Ok(t)
}

/// Mean next-token cross-entropy over all positions and the gradient of every parameter
/// (returned as a [`Gpt2`] of the same shape). `ids[i]` is predicted to be followed by
/// `targets[i]`. `threads` only splits matmuls, so the result is thread-count independent.
pub fn loss_and_grads(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
) -> Result<(f32, Gpt2), String> {
    loss_and_grads_with(m, ids, targets, batch, threads, None)
}
