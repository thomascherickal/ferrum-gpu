use super::*;

pub(super) fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

pub(super) struct DevBlock {
    pub(super) attn_norm: GpuBuf,
    pub(super) wq: Lin,
    pub(super) bq: Option<GpuBuf>,
    pub(super) wk: Lin,
    pub(super) bk: Option<GpuBuf>,
    pub(super) wv: Lin,
    pub(super) bv: Option<GpuBuf>,
    pub(super) wo: Lin,
    pub(super) ffn_norm: GpuBuf,
    pub(super) gate: Lin,
    pub(super) up: Lin,
    pub(super) down: Lin,
    pub(super) kcache: GpuBuf,
    pub(super) vcache: GpuBuf,
}

/// Decode-time scratch (reused every token).
pub(super) struct Scratch {
    pub(super) ids: GpuBuf,
    pub(super) x: GpuBuf,
    pub(super) x2: GpuBuf,
    pub(super) h: GpuBuf,
    pub(super) q: GpuBuf,
    pub(super) k: GpuBuf,
    pub(super) v: GpuBuf,
    pub(super) attn: GpuBuf,
    pub(super) o: GpuBuf,
    pub(super) gate: GpuBuf,
    pub(super) up: GpuBuf,
    pub(super) act: GpuBuf,
    pub(super) f2: GpuBuf,
    pub(super) probs: GpuBuf,
    pub(super) logits: GpuBuf,
    /// Exact-size product buffers for the biased q/k/v projections (kernels need exact lengths).
    pub(super) tmp_q: GpuBuf,
    pub(super) tmp_kv: GpuBuf,
}

/// A Llama-family model resident on the GPU with a KV cache.
pub struct GpuLlama<'g> {
    pub(super) gpu: &'g Gpu,
    pub(super) cfg: Config,
    pub(super) quant: Option<QKind>,
    /// The binding limit this model was built for (the adapter's, unless lowered for a test).
    pub(super) limit: u64,
    pub(super) tok: Embedding,
    pub(super) cos: GpuBuf,
    pub(super) sin: GpuBuf,
    pub(super) blocks: Vec<DevBlock>,
    pub(super) out_norm: GpuBuf,
    pub(super) head: Lin,
    pub(super) zeros: Vec<(usize, GpuBuf)>,
    pub(super) s: Scratch,
    pub(super) matmul: Matmul,
    pub(super) pos: usize,
}

pub(super) fn f32s(n: usize) -> u64 {
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

/// Like [`plan`], but a weight or the embedding larger than `limit` bytes is listed as the blocks
/// it will be stored in (`name.0`, `name.1`, ...), so the binding check sees what is really bound.
/// Buffers that cannot be split (KV cache, scratch) stay whole and fail the check as before.
pub fn plan_limited(c: &Config, quant: Option<QKind>, limit: u64) -> BufferPlan {
    let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
    let mut p = BufferPlan::default();
    let add = |p: &mut BufferPlan, name: &str, bytes: u64, units: usize| {
        let unit = if units == 0 {
            bytes
        } else {
            bytes / units as u64
        };
        match shard::blocks(units, unit, limit) {
            Ok(b) if b.len() > 1 && shard::splittable(name) => {
                for (i, (_, len)) in b.iter().enumerate() {
                    p.add(&format!("{name}.{i}"), *len as u64 * unit);
                }
            }
            _ => p.add(name, bytes),
        }
    };
    add(&mut p, "tok_emb", f32s(c.vocab * d), c.vocab);
    p.add("rope_cos", f32s(c.ctx * c.rope_dim / 2));
    p.add("rope_sin", f32s(c.ctx * c.rope_dim / 2));
    for i in 0..c.layers {
        for (n, bytes, units) in [
            ("attn_norm", f32s(d), 0),
            ("wq", matrix_bytes(quant, d, qd), qd),
            ("wk", matrix_bytes(quant, d, kvd), kvd),
            ("wv", matrix_bytes(quant, d, kvd), kvd),
            ("wo", matrix_bytes(quant, qd, d), d),
            ("ffn_norm", f32s(d), 0),
            ("gate", matrix_bytes(quant, d, h), h),
            ("up", matrix_bytes(quant, d, h), h),
            ("down", matrix_bytes(quant, h, d), d),
            ("kcache", f32s(c.ctx * kvd), 0),
            ("vcache", f32s(c.ctx * kvd), 0),
        ] {
            add(&mut p, &format!("blk{i}.{n}"), bytes, units);
        }
    }
    p.add("out_norm", f32s(d));
    add(&mut p, "head", matrix_bytes(quant, d, c.vocab), c.vocab);
    for (n, len) in [
        ("zero_q", qd),
        ("zero_kv", kvd),
        ("zero_d", d),
        ("zero_h", h),
        ("zero_vocab", c.vocab),
        ("ids", 1),
        ("x", d),
        ("x2", d),
        ("h", d),
        ("q", qd),
        ("k", kvd),
        ("v", kvd),
        ("attn", qd),
        ("o", d),
        ("gate", h),
        ("up", h),
        ("act", h),
        ("f2", d),
        ("probs", c.heads * c.ctx),
        ("logits", c.vocab),
        ("tmp_q", qd),
        ("tmp_kv", kvd),
    ] {
        p.add(&format!("scratch.{n}"), f32s(len));
    }
    p
}
