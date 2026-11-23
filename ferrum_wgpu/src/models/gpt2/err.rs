use super::*;

pub(crate) use crate::device::err;

pub(crate) struct DevBlock {
    pub(crate) ln1_g: GpuBuf,
    pub(crate) ln1_b: GpuBuf,
    pub(crate) wq: GpuBuf,
    pub(crate) bq: GpuBuf,
    pub(crate) wk: GpuBuf,
    pub(crate) bk: GpuBuf,
    pub(crate) wv: GpuBuf,
    pub(crate) bv: GpuBuf,
    pub(crate) wo: GpuBuf,
    pub(crate) bo: GpuBuf,
    pub(crate) ln2_g: GpuBuf,
    pub(crate) ln2_b: GpuBuf,
    pub(crate) w1: GpuBuf,
    pub(crate) b1: GpuBuf,
    pub(crate) w2: GpuBuf,
    pub(crate) b2: GpuBuf,
    pub(crate) kcache: GpuBuf,
    pub(crate) vcache: GpuBuf,
}

/// Decode-time scratch (reused every token).
pub(crate) struct Scratch {
    pub(crate) ids: GpuBuf,
    pub(crate) x: GpuBuf,
    pub(crate) x2: GpuBuf,
    pub(crate) h: GpuBuf,
    pub(crate) q: GpuBuf,
    pub(crate) k: GpuBuf,
    pub(crate) v: GpuBuf,
    pub(crate) attn: GpuBuf,
    pub(crate) o: GpuBuf,
    pub(crate) f1: GpuBuf,
    pub(crate) act: GpuBuf,
    pub(crate) f2: GpuBuf,
    pub(crate) probs: GpuBuf,
    pub(crate) logits: GpuBuf,
}

/// A GPT-2 model resident on the GPU with a KV cache.
pub struct GpuGpt2<'g> {
    pub(crate) gpu: &'g Gpu,
    pub(crate) cfg: Config,
    pub(crate) tok: GpuBuf,
    pub(crate) pos_tab: GpuBuf,
    pub(crate) blocks: Vec<DevBlock>,
    pub(crate) lnf_g: GpuBuf,
    pub(crate) lnf_b: GpuBuf,
    pub(crate) head: GpuBuf,
    pub(crate) zero_vocab: GpuBuf,
    pub(crate) s: Scratch,
    pub(crate) matmul: Matmul,
    pub(crate) pos: usize,
}

pub(crate) fn f32s(n: usize) -> u64 {
    (n * 4) as u64
}
