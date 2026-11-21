use super::*;

pub(super) fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

/// Everything needed to continue a run: weights, Adam moments, steps taken.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaSnapshot {
    /// Weights.
    pub weights: Llama,
    /// First moments.
    pub m: Llama,
    /// Second moments.
    pub v: Llama,
    /// Optimizer steps completed.
    pub steps: u64,
}

/// A Llama-family model being fine-tuned on the GPU.
pub struct GpuLlamaTrainer<'g> {
    pub(super) gpu: &'g Gpu,
    pub(super) cfg: ferrum_ref::llama::Config,
    /// The model's shape with every parameter emptied (read-backs fill it).
    pub(super) template: Llama,
    pub(super) budget: Option<Budget>,
    pub(super) params: Vec<GpuBuf>,
    pub(super) grads: Vec<GpuBuf>,
    pub(super) m: Vec<GpuBuf>,
    pub(super) v: Vec<GpuBuf>,
    pub(super) decays: Vec<bool>,
    pub(super) part: GpuBuf,
    pub(super) part_offsets: Vec<usize>,
    pub(super) scale: GpuBuf,
    pub(super) matmul: Matmul,
    pub(super) acts: Option<Acts>,
    pub(super) t: u64,
}
