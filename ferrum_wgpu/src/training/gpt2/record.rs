//! The forward and backward dispatch sequence of one training step.
use super::GpuTrainer;
use crate::device::GpuError;
use crate::tensor::GpuBuf;
mod backward;
mod forward;
mod param_index;
use param_index::*;

/// The sizes both passes use: `r = batch * t` rows, width `d`, feed-forward width `h`.
struct Dims {
    b: usize,
    t: usize,
    nl: usize,
    r: usize,
    d: usize,
    h: usize,
    v: usize,
    hd: usize,
}

impl GpuTrainer<'_> {
    /// Record the forward and backward dispatches (a batch must be open).
    pub(crate) fn record(
        &self,
        ids_host: &[usize],
        ids: &GpuBuf,
        targets: &GpuBuf,
    ) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let c = &self.cfg;
        let (b, t) = (a.batch, a.t);
        let dims = Dims {
            b,
            t,
            nl: c.layers,
            r: b * t,
            d: c.dim,
            h: c.hidden,
            v: c.vocab,
            hd: c.head_dim(),
        };
        self.record_forward(&dims, ids, targets)?;
        self.record_backward(&dims, ids_host)
    }
}
