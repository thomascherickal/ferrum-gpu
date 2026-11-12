//! The dropout kernel: mask a tensor in place with the counter-hash mask of
//! [`ferrum_ref::dropout`], so GPU and CPU drop exactly the same units.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::dropout::{keep_scale, Dropout};

/// Where a tensor sits in the mask: the block (layer) it belongs to, and the global index of its
/// first unit (non-zero for the second and later micro-batches of a step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    /// Layer index; masks differ per block.
    pub block: u32,
    /// Global unit index of `x[0]`.
    pub offset: u32,
}

/// `x[i] = 0` if unit `place.offset + i` is dropped under `d`, else `x[i] * 1/(1-rate)`.
pub fn dropout_inplace(gpu: &Gpu, x: &GpuBuf, d: &Dropout, place: Place) -> Result<(), GpuError> {
    if u32::try_from(x.len())
        .ok()
        .and_then(|n| n.checked_add(place.offset))
        .is_none()
    {
        return Err(GpuError("dropout: unit indices exceed 32 bits".into()));
    }
    let (lo, hi) = d.seed_words();
    let p = pipe!(gpu, "fwd/dropout")?;
    gpu.dispatch_1d(
        &p,
        &[x],
        &[
            x.len() as u32,
            d.rate.to_bits(),
            keep_scale(d.rate).to_bits(),
            lo,
            hi,
            place.block,
            place.offset,
            0,
        ],
        x.len(),
    )
}
