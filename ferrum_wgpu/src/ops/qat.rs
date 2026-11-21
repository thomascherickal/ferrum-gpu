//! Quantisation-aware training kernel: snap a tensor onto its Q8_0 values on the device.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::qat::eligible;

fn check(src: &GpuBuf, dst: &GpuBuf) -> Result<(), GpuError> {
    if src.len() == dst.len() {
        Ok(())
    } else {
        Err(GpuError(format!(
            "q8_snap: source holds {} elements but destination {}",
            src.len(),
            dst.len()
        )))
    }
}

/// Snap `src` into `dst` along rows: blocks are 32 consecutive elements and `ne0` (the row
/// length) must be a multiple of 32. An ineligible `ne0` copies the data unchanged.
pub fn q8_snap_rows(gpu: &Gpu, src: &GpuBuf, dst: &GpuBuf, ne0: usize) -> Result<(), GpuError> {
    check(src, dst)?;
    if !eligible(ne0) || !src.len().is_multiple_of(32) {
        return gpu.copy_f32(src, 0, dst, 0, src.len());
    }
    let p = pipe!(gpu, "quant/q8_snap")?;
    let blocks = src.len() / 32;
    gpu.dispatch_1d(&p, &[src, dst], &[blocks as u32, 0, 0], blocks)
}

/// Snap an `[rows_in, cols_out]` row-major weight down its columns (GGUF's layout). An
/// ineligible `rows_in` (not a multiple of 32) copies the data unchanged.
pub fn q8_snap_cols(
    gpu: &Gpu,
    src: &GpuBuf,
    dst: &GpuBuf,
    (rows_in, cols_out): (usize, usize),
) -> Result<(), GpuError> {
    check(src, dst)?;
    if src.len() != rows_in * cols_out {
        return Err(GpuError(format!(
            "q8_snap: {} elements is not {rows_in} x {cols_out}",
            src.len()
        )));
    }
    if !eligible(rows_in) {
        return gpu.copy_f32(src, 0, dst, 0, src.len());
    }
    let p = pipe!(gpu, "quant/q8_snap")?;
    let blocks = rows_in / 32 * cols_out;
    gpu.dispatch_1d(
        &p,
        &[src, dst],
        &[blocks as u32, 1, cols_out as u32],
        blocks,
    )
}
#[cfg(test)]
mod tests;
