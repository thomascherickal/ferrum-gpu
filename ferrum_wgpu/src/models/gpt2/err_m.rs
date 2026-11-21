use super::*;

pub(super) fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

pub(super) struct DevBlock {
    pub(super) ln1_g: GpuBuf,
    pub(super) ln1_b: GpuBuf,
    pub(super) wq: GpuBuf,
    pub(super) bq: GpuBuf,
    pub(super) wk: GpuBuf,
    pub(super) bk: GpuBuf,
    pub(super) wv: GpuBuf,
    pub(super) bv: GpuBuf,
    pub(super) wo: GpuBuf,
    pub(super) bo: GpuBuf,
    pub(super) ln2_g: GpuBuf,
    pub(super) ln2_b: GpuBuf,
    pub(super) w1: GpuBuf,
    pub(super) b1: GpuBuf,
    pub(super) w2: GpuBuf,
    pub(super) b2: GpuBuf,
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
    pub(super) f1: GpuBuf,
    pub(super) act: GpuBuf,
    pub(super) f2: GpuBuf,
    pub(super) probs: GpuBuf,
    pub(super) logits: GpuBuf,
}

/// A GPT-2 model resident on the GPU with a KV cache.
pub struct GpuGpt2<'g> {
    pub(super) gpu: &'g Gpu,
    pub(super) cfg: Config,
    pub(super) tok: GpuBuf,
    pub(super) pos_tab: GpuBuf,
    pub(super) blocks: Vec<DevBlock>,
    pub(super) lnf_g: GpuBuf,
    pub(super) lnf_b: GpuBuf,
    pub(super) head: GpuBuf,
    pub(super) zero_vocab: GpuBuf,
    pub(super) s: Scratch,
    pub(super) matmul: Matmul,
    pub(super) pos: usize,
}

pub(super) fn f32s(n: usize) -> u64 {
    (n * 4) as u64
}

/// Every device buffer a model of this shape needs, named and sized (weights, KV cache, scratch).
pub fn plan(c: &Config) -> BufferPlan {
    let (d, h) = (c.dim, c.hidden);
    let mut p = BufferPlan::default();
    p.add("tok_emb", f32s(c.vocab * d));
    p.add("pos_emb", f32s(c.ctx * d));
    for i in 0..c.layers {
        for (n, len) in [
            ("ln1_g", d),
            ("ln1_b", d),
            ("wq", d * d),
            ("bq", d),
            ("wk", d * d),
            ("bk", d),
            ("wv", d * d),
            ("bv", d),
            ("wo", d * d),
            ("bo", d),
            ("ln2_g", d),
            ("ln2_b", d),
            ("w1", d * h),
            ("b1", h),
            ("w2", h * d),
            ("b2", d),
            ("kcache", c.ctx * d),
            ("vcache", c.ctx * d),
        ] {
            p.add(&format!("blk{i}.{n}"), f32s(len));
        }
    }
    p.add("lnf_g", f32s(d));
    p.add("lnf_b", f32s(d));
    p.add("head", f32s(d * c.vocab));
    p.add("zero_vocab", f32s(c.vocab));
    for (n, len) in [
        ("ids", 1),
        ("x", d),
        ("x2", d),
        ("h", d),
        ("q", d),
        ("k", d),
        ("v", d),
        ("attn", d),
        ("o", d),
        ("f1", h),
        ("act", h),
        ("f2", d),
        ("probs", c.heads * c.ctx),
        ("logits", c.vocab),
    ] {
        p.add(&format!("scratch.{n}"), f32s(len));
    }
    p
}
