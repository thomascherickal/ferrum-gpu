use super::{err, need};
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// Softmax cross-entropy: fills `grad` (`rows x vocab`) and the one-element `loss` (mean).
/// `row_loss` is scratch of `rows` elements; `targets` come from [`Gpu::upload_indices`].
pub fn cross_entropy(
    gpu: &Gpu,
    logits: &GpuBuf,
    targets: &GpuBuf,
    (grad, row_loss, loss): (&GpuBuf, &GpuBuf, &GpuBuf),
    (rows, vocab): (usize, usize),
) -> Result<(), GpuError> {
    need("cross_entropy logits", logits, rows * vocab)?;
    need("cross_entropy targets", targets, rows)?;
    need("cross_entropy grad", grad, rows * vocab)?;
    need("cross_entropy row_loss", row_loss, rows)?;
    need("cross_entropy loss", loss, 1)?;
    if rows == 0 {
        return Err(err("cross_entropy: zero rows"));
    }
    let p1 = pipe!(gpu, "fwd/ce_rows")?;
    gpu.dispatch_1d(
        &p1,
        &[logits, targets, grad, row_loss],
        &[rows as u32, vocab as u32],
        rows,
    )?;
    let p2 = pipe!(gpu, "fwd/mean_loss")?;
    gpu.dispatch(&p2, &[row_loss, loss], &[rows as u32], (1, 1))
}

/// Per-row `(log-probability of target, 1.0 if target is the first argmax else 0.0)` into `out` (`2*rows`).
pub fn score(
    gpu: &Gpu,
    logits: &GpuBuf,
    targets: &GpuBuf,
    out: &GpuBuf,
    (rows, vocab): (usize, usize),
) -> Result<(), GpuError> {
    need("score logits", logits, rows * vocab)?;
    need("score targets", targets, rows)?;
    need("score out", out, 2 * rows)?;
    let p = pipe!(gpu, "fwd/score")?;
    gpu.dispatch_1d(
        &p,
        &[logits, targets, out],
        &[rows as u32, vocab as u32],
        rows,
    )
}
