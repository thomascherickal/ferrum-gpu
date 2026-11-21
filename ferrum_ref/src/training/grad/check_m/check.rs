use super::*;

pub(crate) fn check(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
) -> Result<usize, String> {
    let c = &m.cfg;
    if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
        return Err(format!(
            "{} tokens do not split into {batch} sequences",
            ids.len()
        ));
    }
    let t = ids.len() / batch;
    if t > c.ctx {
        return Err(format!("sequence length {t} exceeds the context {}", c.ctx));
    }
    if targets.len() != ids.len() {
        return Err(format!(
            "{} targets for {} tokens",
            targets.len(),
            ids.len()
        ));
    }
    if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= c.vocab) {
        return Err(format!("token {bad} is out of range for vocab {}", c.vocab));
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
