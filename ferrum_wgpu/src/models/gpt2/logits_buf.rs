use super::*;

impl<'g> GpuGpt2<'g> {
    /// Like [`forward_all`](Self::forward_all) but leaves the logits on the device.
    pub(super) fn forward_logits_buf(&self, ids: &[usize]) -> Result<GpuBuf, GpuError> {
        let host = self.forward_all(ids)?;
        // ponytail: forward_all reads back; re-upload keeps score simple. Return the buffer
        // directly once forward_all is split into a resident part and a readback part.
        self.gpu.upload(&host)
    }
}
