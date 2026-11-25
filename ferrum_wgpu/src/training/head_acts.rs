//! The loss-head activations both trainers keep: final norm output, logits, their gradient, the
//! per-row and mean loss, and the gradient into the final norm.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// Loss-head buffers for `r` rows of width `d` over a vocabulary of `v`.
pub(crate) struct HeadActs {
    pub(crate) hf: GpuBuf,
    pub(crate) logits: GpuBuf,
    pub(crate) dlogits: GpuBuf,
    pub(crate) row_loss: GpuBuf,
    pub(crate) loss: GpuBuf,
    pub(crate) dhf: GpuBuf,
}

impl HeadActs {
    /// `(name, f32 count)` of each buffer, for the footprint plan.
    pub(crate) fn rows(r: usize, d: usize, v: usize) -> [(&'static str, usize); 6] {
        let names = ["hf", "logits", "dlogits", "row_loss", "loss", "dhf"];
        let lens = [r * d, r * v, r * v, r, 1, r * d];
        std::array::from_fn(|i| (names[i], lens[i]))
    }

    /// Zeroed buffers, sized as [`rows`](Self::rows) says.
    pub(crate) fn new(gpu: &Gpu, r: usize, d: usize, v: usize) -> Result<Self, GpuError> {
        let [hf, logits, dlogits, row_loss, loss, dhf] =
            Self::rows(r, d, v).map(|(_, n)| gpu.zeros(n));
        Ok(HeadActs {
            hf: hf?,
            logits: logits?,
            dlogits: dlogits?,
            row_loss: row_loss?,
            loss: loss?,
            dhf: dhf?,
        })
    }
}
