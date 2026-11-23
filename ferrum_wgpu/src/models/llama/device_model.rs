use super::*;

pub(crate) fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

pub(crate) struct DevBlock {
    pub(crate) attn_norm: GpuBuf,
    pub(crate) wq: Lin,
    pub(crate) bq: Option<GpuBuf>,
    pub(crate) wk: Lin,
    pub(crate) bk: Option<GpuBuf>,
    pub(crate) wv: Lin,
    pub(crate) bv: Option<GpuBuf>,
    pub(crate) wo: Lin,
    pub(crate) ffn_norm: GpuBuf,
    pub(crate) gate: Lin,
    pub(crate) up: Lin,
    pub(crate) down: Lin,
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
    pub(crate) gate: GpuBuf,
    pub(crate) up: GpuBuf,
    pub(crate) act: GpuBuf,
    pub(crate) f2: GpuBuf,
    pub(crate) probs: GpuBuf,
    pub(crate) logits: GpuBuf,
    /// Exact-size product buffers for the biased q/k/v projections (kernels need exact lengths).
    pub(crate) tmp_q: GpuBuf,
    pub(crate) tmp_kv: GpuBuf,
}

/// A Llama-family model resident on the GPU with a KV cache.
pub struct GpuLlama<'g> {
    pub(crate) gpu: &'g Gpu,
    pub(crate) cfg: Config,
    pub(crate) quant: Option<QKind>,
    /// The binding limit this model was built for (the adapter's, unless lowered for a test).
    pub(crate) limit: u64,
    pub(crate) tok: Embedding,
    pub(crate) cos: GpuBuf,
    pub(crate) sin: GpuBuf,
    pub(crate) blocks: Vec<DevBlock>,
    pub(crate) out_norm: GpuBuf,
    pub(crate) head: Lin,
    pub(crate) zeros: Vec<(usize, GpuBuf)>,
    pub(crate) s: Scratch,
    pub(crate) matmul: Matmul,
    pub(crate) pos: usize,
}

pub(crate) fn f32s(n: usize) -> u64 {
    (n * 4) as u64
}

/// Bytes a `k x n` matrix occupies on the device under `quant`.
pub(super) fn matrix_bytes(quant: Option<QKind>, k: usize, n: usize) -> u64 {
    quant
        .map(|q| effective(q, k))
        .and_then(|q| packed_words(q, k, n))
        .map_or(f32s(k * n), f32s)
}

/// Every device buffer a model of this shape needs, named and sized (weights, RoPE tables, KV
/// cache, scratch).
pub fn plan(c: &Config, quant: Option<QKind>) -> BufferPlan {
    plan_limited(c, quant, u64::MAX)
}
