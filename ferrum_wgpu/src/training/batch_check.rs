//! The shape checks both trainers make before a step.

use crate::device::{err, GpuError};

/// Check that `ids` split into `batch` sequences that fit `ctx`, that `targets` match them and that every
/// token is below `vocab`; returns the sequence length.
pub(crate) fn check_batch(
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    ctx: usize,
    vocab: usize,
) -> Result<usize, GpuError> {
    if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
        return Err(err(format!(
            "{} tokens do not split into {batch} sequences",
            ids.len()
        )));
    }
    let t = ids.len() / batch;
    if t > ctx {
        return Err(err(format!(
            "sequence length {t} exceeds the context {ctx}"
        )));
    }
    if targets.len() != ids.len() {
        return Err(err(format!(
            "{} targets for {} tokens",
            targets.len(),
            ids.len()
        )));
    }
    match ids.iter().chain(targets).find(|&&i| i >= vocab) {
        Some(bad) => Err(err(format!(
            "token {bad} is out of range for vocab {vocab}"
        ))),
        None => Ok(t),
    }
}
