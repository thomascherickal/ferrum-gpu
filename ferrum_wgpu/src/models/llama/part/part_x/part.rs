use super::*;

impl<'g> GpuLlama<'g> {
    /// Upload `w`, quantising eligible matrices to `quant` when given. The buffer plan is checked
    /// against the adapter and `budget` first, so an oversized model is refused before any device
    /// memory is allocated.
    pub fn new(
        gpu: &'g Gpu,
        w: &Llama,
        quant: Option<QKind>,
        budget: Option<Budget>,
    ) -> Result<Self, GpuError> {
        Self::with_binding_limit(gpu, w, quant, budget, gpu.info().max_binding)
    }
}
