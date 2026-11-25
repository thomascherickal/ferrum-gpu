use super::*;

pub(super) use crate::device::err;

/// A GPT-2 model being trained on the GPU.
pub struct GpuTrainer<'g> {
    pub(super) gpu: &'g Gpu,
    pub(super) cfg: Config,
    pub(super) budget: Option<Budget>,
    pub(super) state: crate::training::DeviceParams,
    pub(super) matmul: Matmul,
    pub(super) acts: Option<Acts>,
    pub(super) accum: Option<Vec<GpuBuf>>,
    pub(super) last_micro: usize,
    pub(super) qat: bool,
    /// The head is tied to the embedding (see `tie`).
    pub(super) tie: bool,
    /// FFN-hidden dropout rate and the run's base seed (see `dropout`).
    pub(super) dropout: (f32, u64),
    /// Global index of the first FFN unit of the batch being recorded (micro-batch offset).
    pub(super) drop_offset: Cell<u32>,
    pub(super) fwd: Option<Vec<GpuBuf>>,
    pub(super) t: u64,
}
