//! The forward and backward dispatch sequence of one training step.
use super::GpuTrainer;
use crate::device::GpuError;
use crate::tensor::GpuBuf;
use crate::training::Dims;
mod backward;
mod forward;
mod param_index;
use param_index::*;

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
        let widths = [c.dim, c.hidden, c.vocab, c.head_dim(), c.dim, c.dim];
        let dims = Dims::new((a.batch, a.t), c.layers, widths);
        self.record_forward(&dims, ids, targets)?;
        self.record_backward(&dims, ids_host)
    }
}
