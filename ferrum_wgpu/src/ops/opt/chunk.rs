use super::*;

/// Elements summarised by one workgroup of the sum-of-squares kernel.
pub const CHUNK: usize = 4096;

/// Partial sums a tensor of `n` elements produces.
pub fn parts_for(n: usize) -> usize {
    n.div_ceil(CHUNK).max(1)
}

/// The sum-of-squares kernel runs one workgroup per partial along x, so `partials` must fit.
/// (Split out so the limit is testable without a 1 GB tensor.)
pub(super) fn check_dispatchable(partials: usize) -> Result<(), GpuError> {
    if partials > crate::runtime::GROUPS_X {
        return Err(GpuError(
            "sumsq_parts: tensor too large for one dispatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
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
