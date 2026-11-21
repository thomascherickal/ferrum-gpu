//! Optimizer kernels: deterministic global gradient norm, clip scale, and the AdamW update.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::optim::Adam;

/// Elements summarised by one workgroup of the sum-of-squares kernel.
pub const CHUNK: usize = 4096;

/// Partial sums a tensor of `n` elements produces.
pub fn parts_for(n: usize) -> usize {
    n.div_ceil(CHUNK).max(1)
}

/// The sum-of-squares kernel runs one workgroup per partial along x, so `partials` must fit.
/// (Split out so the limit is testable without a 1 GB tensor.)
fn check_dispatchable(partials: usize) -> Result<(), GpuError> {
    if partials > crate::runtime::GROUPS_X {
        return Err(GpuError(
            "sumsq_parts: tensor too large for one dispatch".into(),
        ));
    }
    Ok(())
}

fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(GpuError(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

/// Write `parts_for(g.len())` partial sums of squares of `g` into `part[offset..]`.
pub fn sumsq_parts(gpu: &Gpu, g: &GpuBuf, part: &GpuBuf, offset: usize) -> Result<(), GpuError> {
    let k = parts_for(g.len());
    if offset + k > part.len() {
        return Err(GpuError(format!(
            "sumsq_parts: {k} partials at {offset} exceed the buffer of {}",
            part.len()
        )));
    }
    check_dispatchable(k)?;
    let p = pipe!(gpu, "opt/sumsq_part")?;
    gpu.dispatch(
        &p,
        &[g, part],
        &[g.len() as u32, offset as u32],
        (k as u32, 1),
    )
}

/// `out = [clip scale, global norm]` from the first `total` partials.
pub fn norm_scale(
    gpu: &Gpu,
    part: &GpuBuf,
    out: &GpuBuf,
    total: usize,
    max_norm: f32,
) -> Result<(), GpuError> {
    need("norm_scale out", out, 2)?;
    if total > part.len() {
        return Err(GpuError(format!(
            "norm_scale: {total} partials exceed the buffer of {}",
            part.len()
        )));
    }
    let p = pipe!(gpu, "opt/norm_final")?;
    gpu.dispatch(
        &p,
        &[part, out],
        &[total as u32, max_norm.to_bits()],
        (1, 1),
    )
}

/// `acc = (first ? 0 : acc) + weight * g`: add one micro-batch's gradient into the running sum.
pub fn accumulate(
    gpu: &Gpu,
    acc: &GpuBuf,
    g: &GpuBuf,
    weight: f32,
    first: bool,
) -> Result<(), GpuError> {
    need("accumulate g", g, acc.len())?;
    let p = pipe!(gpu, "opt/accumulate")?;
    gpu.dispatch_1d(
        &p,
        &[acc, g],
        &[acc.len() as u32, weight.to_bits(), u32::from(first)],
        acc.len(),
    )
}

/// `dst[r][c] = (add ? dst[r][c] : 0) + src[c][r]`: `dst` is `[rows, cols]` and `src` is its
/// transpose `[cols, rows]`. Ties a head to an embedding (`add = false`) and folds the head's
/// gradient into the embedding's (`add = true`).
pub fn transpose_into(
    gpu: &Gpu,
    src: &GpuBuf,
    dst: &GpuBuf,
    (rows, cols): (usize, usize),
    add: bool,
) -> Result<(), GpuError> {
    need("transpose dst", dst, rows * cols)?;
    need("transpose src", src, rows * cols)?;
    let p = pipe!(gpu, "opt/transpose")?;
    gpu.dispatch_1d(
        &p,
        &[dst, src],
        &[rows as u32, cols as u32, u32::from(add), 0],
        rows * cols,
    )
}

/// One AdamW update of a tensor. `decay` selects whether weight decay applies; `t` is the
/// 1-based step (bias corrections are computed here, on the host). `scale` is `[clip scale, ..]`.
pub fn adam_update(
    gpu: &Gpu,
    (p, g, m, v): (&GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf),
    scale: &GpuBuf,
    adam: &Adam,
    decay: bool,
    t: u64,
) -> Result<(), GpuError> {
    let n = p.len();
    for (name, b) in [("grad", g), ("m", m), ("v", v)] {
        need(&format!("adam_update {name}"), b, n)?;
    }
    if t == 0 {
        return Err(GpuError("adam timestep must be at least 1".into()));
    }
    if scale.is_empty() {
        return Err(GpuError("adam_update: scale buffer is empty".into()));
    }
    let (bc1, bc2) = adam.bias_corrections(t);
    let wd = if decay { adam.weight_decay } else { 0.0 };
    let u = [
        n as u32,
        adam.lr.to_bits(),
        adam.beta1.to_bits(),
        adam.beta2.to_bits(),
        adam.eps.to_bits(),
        wd.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let pipe = pipe!(gpu, "opt/adam")?;
    gpu.dispatch_1d(&pipe, &[p, g, m, v, scale], &u, n)
}
#[cfg(test)]
mod tests;
