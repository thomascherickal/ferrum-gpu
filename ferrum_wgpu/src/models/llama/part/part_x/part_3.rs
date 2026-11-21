use super::*;

impl<'g> GpuLlama<'g> {
    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The largest buffer binding, in bytes, this model was built for: the adapter's own limit
    /// unless [`with_binding_limit`](Self::with_binding_limit) lowered it.
    pub fn binding_limit(&self) -> u64 {
        self.limit
    }

    /// The quantisation requested at upload (ineligible matrices stay f32).
    pub fn quant(&self) -> Option<QKind> {
        self.quant
    }

    /// Tokens consumed since the last [`reset`](Self::reset).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Forget the cached context (the cache buffers are overwritten as new tokens arrive).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub(crate) fn zero(&self, n: usize) -> &GpuBuf {
        &self
            .zeros
            .iter()
            .find(|(m, _)| *m == n)
            .expect("a zero bias exists for every projection width")
            .1
    }
}
