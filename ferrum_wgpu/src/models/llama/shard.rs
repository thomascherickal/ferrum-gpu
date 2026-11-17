//! Splitting a tensor that is larger than one storage binding into blocks that each fit.
//!
//! A weight `[k, n]` is cut into column blocks (each block is an ordinary `[k, width]` matrix that
//! quantises and multiplies on its own) and the token embedding `[vocab, d]` into row blocks. This
//! file is only arithmetic, so it is tested without a device.

use crate::device::GpuError;

/// `(start, len)` blocks of `total` units, each at most `limit` bytes when one unit is `unit_bytes`.
/// One block if everything fits. An error if a single unit is already larger than the limit.
pub(super) fn blocks(
    total: usize,
    unit_bytes: u64,
    limit: u64,
) -> Result<Vec<(usize, usize)>, GpuError> {
    if total == 0 || unit_bytes == 0 || (total as u64).saturating_mul(unit_bytes) <= limit {
        return Ok(vec![(0, total)]);
    }
    let per = limit / unit_bytes;
    if per == 0 {
        return Err(GpuError(format!(
            "one row or column of {unit_bytes} bytes is larger than the binding limit of {limit} bytes, so it cannot be split further"
        )));
    }
    let per = per as usize;
    Ok((0..total)
        .step_by(per)
        .map(|s| (s, per.min(total - s)))
        .collect())
}

/// The columns `[c0, c0 + width)` of a row-major `[k, n]` matrix, as a row-major `[k, width]` one.
pub(super) fn column_block(
    w: &[f32],
    (k, n): (usize, usize),
    (c0, width): (usize, usize),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(k * width);
    for r in 0..k {
        out.extend_from_slice(&w[r * n + c0..r * n + c0 + width]);
    }
    out
}

/// Whether `name` (as listed in a buffer plan) is a tensor that can be split into blocks. The KV
/// cache, scratch and the prefill logits are single buffers and are not.
pub(super) fn splittable(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    matches!(
        leaf,
        "tok_emb" | "head" | "wq" | "wk" | "wv" | "wo" | "gate" | "up" | "down"
    ) && !name.starts_with("scratch.")
}

#[cfg(test)]
mod tests;
