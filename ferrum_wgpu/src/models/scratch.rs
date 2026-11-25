//! Decode-time scratch shared by the GPT-2 and Llama device models.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// Widths of the decode scratch: model `d`, query `qd`, key/value `kvd`, feed-forward `h`, and
/// the `heads * ctx` attention probabilities and `vocab` logits.
pub(crate) struct ScratchShape {
    pub(crate) d: usize,
    pub(crate) qd: usize,
    pub(crate) kvd: usize,
    pub(crate) h: usize,
    pub(crate) probs: usize,
    pub(crate) vocab: usize,
}

/// Decode-time scratch (reused every token); `extra` holds the model's own buffers.
pub(crate) struct Scratch<E> {
    pub(crate) ids: GpuBuf,
    pub(crate) x: GpuBuf,
    pub(crate) x2: GpuBuf,
    pub(crate) h: GpuBuf,
    pub(crate) q: GpuBuf,
    pub(crate) k: GpuBuf,
    pub(crate) v: GpuBuf,
    pub(crate) attn: GpuBuf,
    pub(crate) o: GpuBuf,
    pub(crate) act: GpuBuf,
    pub(crate) f2: GpuBuf,
    pub(crate) probs: GpuBuf,
    pub(crate) logits: GpuBuf,
    pub(crate) extra: E,
}

impl<E> Scratch<E> {
    /// Zeroed scratch of shape `s` around the model's `extra` buffers.
    pub(crate) fn new(gpu: &Gpu, s: ScratchShape, extra: E) -> Result<Self, GpuError> {
        let z = |n| gpu.zeros(n);
        Ok(Scratch {
            ids: z(1)?,
            x: z(s.d)?,
            x2: z(s.d)?,
            h: z(s.d)?,
            q: z(s.qd)?,
            k: z(s.kvd)?,
            v: z(s.kvd)?,
            attn: z(s.qd)?,
            o: z(s.d)?,
            act: z(s.h)?,
            f2: z(s.d)?,
            probs: z(s.probs)?,
            logits: z(s.vocab)?,
            extra,
        })
    }
}
