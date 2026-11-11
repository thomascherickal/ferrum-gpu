//! Llama / Qwen2 inference on the GPU with every weight, the KV cache and all activations
//! resident. Matrices are f32 or Q8_0/Q4_0 packed (dequantised in the kernel). Per token the host
//! sends one index and receives one logits row. The CPU oracle is `ferrum_ref::llama`.
//! ponytail: single-row decode uses the plain f32 `gemv` or the one-invocation-per-output
//! quantised kernel; split K across the workgroup (as `gemv.wgsl` does) if a profile on a large
//! model says decode is latency-bound.

mod forward;
mod lin;
#[cfg(test)]
mod tests;

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops;
use crate::ops_llama;
use crate::ops_quant::{eligible, packed_words, quantise_matrix, QKind};
use crate::tensor::GpuBuf;
use ferrum_gguf::quant::decode;
use ferrum_ref::llama::{rope_tables, transpose, Config, Llama};
use lin::Lin;

fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

struct DevBlock {
    attn_norm: GpuBuf,
    wq: Lin,
    bq: Option<GpuBuf>,
    wk: Lin,
    bk: Option<GpuBuf>,
    wv: Lin,
    bv: Option<GpuBuf>,
    wo: Lin,
    ffn_norm: GpuBuf,
    gate: Lin,
    up: Lin,
    down: Lin,
    kcache: GpuBuf,
    vcache: GpuBuf,
}

/// Decode-time scratch (reused every token).
struct Scratch {
    ids: GpuBuf,
    x: GpuBuf,
    x2: GpuBuf,
    h: GpuBuf,
    q: GpuBuf,
    k: GpuBuf,
    v: GpuBuf,
    attn: GpuBuf,
    o: GpuBuf,
    gate: GpuBuf,
    up: GpuBuf,
    act: GpuBuf,
    f2: GpuBuf,
    probs: GpuBuf,
    logits: GpuBuf,
    /// Exact-size product buffers for the biased q/k/v projections (kernels need exact lengths).
    tmp_q: GpuBuf,
    tmp_kv: GpuBuf,
}

/// A Llama-family model resident on the GPU with a KV cache.
pub struct GpuLlama<'g> {
    gpu: &'g Gpu,
    cfg: Config,
    quant: Option<QKind>,
    tok: GpuBuf,
    cos: GpuBuf,
    sin: GpuBuf,
    blocks: Vec<DevBlock>,
    out_norm: GpuBuf,
    head: Lin,
    zeros: Vec<(usize, GpuBuf)>,
    s: Scratch,
    matmul: Matmul,
    pos: usize,
}

fn f32s(n: usize) -> u64 {
    (n * 4) as u64
}

/// Bytes a `k x n` matrix occupies on the device under `quant`.
fn matrix_bytes(quant: Option<QKind>, k: usize, n: usize) -> u64 {
    quant
        .and_then(|q| packed_words(q, k, n))
        .map_or(f32s(k * n), f32s)
}

/// Every device buffer a model of this shape needs, named and sized (weights, RoPE tables, KV
/// cache, scratch).
pub fn plan(c: &Config, quant: Option<QKind>) -> BufferPlan {
    let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
    let mut p = BufferPlan::default();
    p.add("tok_emb", f32s(c.vocab * d));
    p.add("rope_cos", f32s(c.ctx * c.rope_dim / 2));
    p.add("rope_sin", f32s(c.ctx * c.rope_dim / 2));
    for i in 0..c.layers {
        for (n, bytes) in [
            ("attn_norm", f32s(d)),
            ("wq", matrix_bytes(quant, d, qd)),
            ("wk", matrix_bytes(quant, d, kvd)),
            ("wv", matrix_bytes(quant, d, kvd)),
            ("wo", matrix_bytes(quant, qd, d)),
            ("ffn_norm", f32s(d)),
            ("gate", matrix_bytes(quant, d, h)),
            ("up", matrix_bytes(quant, d, h)),
            ("down", matrix_bytes(quant, h, d)),
            ("kcache", f32s(c.ctx * kvd)),
            ("vcache", f32s(c.ctx * kvd)),
        ] {
            p.add(&format!("blk{i}.{n}"), bytes);
        }
    }
    p.add("out_norm", f32s(d));
    p.add("head", matrix_bytes(quant, d, c.vocab));
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

/// What the GPU actually computes when `quant` is requested: `m` with every matrix that
/// can be quantised (input width a multiple of 32) replaced by its dequantised value. A model
/// loaded from a Q8_0/Q4_0 GGUF is already in this form (requantising it is the identity), so
/// this is the oracle for a quantised run of an f32 model.
pub fn dequantised(m: &Llama, quant: QKind) -> Result<Llama, GpuError> {
    let snap = |w: &[f32], k: usize, n: usize| -> Result<Vec<f32>, GpuError> {
        if !eligible(k) {
            return Ok(w.to_vec());
        }
        let bytes = quantise_matrix(quant, w, k, n)?;
        let rows = decode(quant_ggml(quant), &bytes, k * n).map_err(|e| err(e.to_string()))?;
        Ok(transpose(&rows, n, k))
    };
    let c = &m.cfg;
    let mut out = m.clone();
    for b in &mut out.blocks {
        b.wq = snap(&b.wq, c.dim, c.q_dim())?;
        b.wk = snap(&b.wk, c.dim, c.kv_dim())?;
        b.wv = snap(&b.wv, c.dim, c.kv_dim())?;
        b.wo = snap(&b.wo, c.q_dim(), c.dim)?;
        b.gate = snap(&b.gate, c.dim, c.hidden)?;
        b.up = snap(&b.up, c.dim, c.hidden)?;
        b.down = snap(&b.down, c.hidden, c.dim)?;
    }
    let head = snap(&m.head, c.dim, c.vocab)?;
    out.tied = out.tied && head == m.head;
    out.head = head;
    Ok(out)
}

fn quant_ggml(q: QKind) -> u32 {
    match q {
        QKind::Q8_0 => ferrum_gguf::GGML_Q8_0,
        QKind::Q4_0 => ferrum_gguf::GGML_Q4_0,
    }
}

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
        w.cfg.validate().map_err(err)?;
        w.rope.check(w.cfg.rope_dim).map_err(err)?;
        plan(&w.cfg, quant)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let c = w.cfg;
        let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
        let up = |v: &[f32]| gpu.upload(v);
        let bias = |v: &[f32]| (!v.is_empty()).then(|| gpu.upload(v)).transpose();
        let lin = |m: &[f32], k, n| Lin::upload(gpu, m, (k, n), quant);
        let blocks = w
            .blocks
            .iter()
            .map(|b| {
                Ok(DevBlock {
                    attn_norm: up(&b.attn_norm)?,
                    wq: lin(&b.wq, d, qd)?,
                    bq: bias(&b.bq)?,
                    wk: lin(&b.wk, d, kvd)?,
                    bk: bias(&b.bk)?,
                    wv: lin(&b.wv, d, kvd)?,
                    bv: bias(&b.bv)?,
                    wo: lin(&b.wo, qd, d)?,
                    ffn_norm: up(&b.ffn_norm)?,
                    gate: lin(&b.gate, d, h)?,
                    up: lin(&b.up, d, h)?,
                    down: lin(&b.down, h, d)?,
                    kcache: gpu.zeros(c.ctx * kvd)?,
                    vcache: gpu.zeros(c.ctx * kvd)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let z = |n| gpu.zeros(n);
        let s = Scratch {
            ids: z(1)?,
            x: z(d)?,
            x2: z(d)?,
            h: z(d)?,
            q: z(qd)?,
            k: z(kvd)?,
            v: z(kvd)?,
            attn: z(qd)?,
            o: z(d)?,
            gate: z(h)?,
            up: z(h)?,
            act: z(h)?,
            f2: z(d)?,
            probs: z(c.heads * c.ctx)?,
            logits: z(c.vocab)?,
            tmp_q: z(qd)?,
            tmp_kv: z(kvd)?,
        };
        let mut zeros: Vec<(usize, GpuBuf)> = Vec::new();
        for n in [qd, kvd, d, h, c.vocab] {
            if !zeros.iter().any(|(m, _)| *m == n) {
                zeros.push((n, z(n)?));
            }
        }
        let (cos, sin) = rope_tables(0, c.ctx, c.rope_dim, c.rope_base, &w.rope);
        Ok(GpuLlama {
            gpu,
            cfg: c,
            quant,
            tok: up(&w.tok)?,
            cos: up(&cos)?,
            sin: up(&sin)?,
            blocks,
            out_norm: up(&w.out_norm)?,
            head: lin(&w.head, d, c.vocab)?,
            zeros,
            s,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            pos: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
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

    fn zero(&self, n: usize) -> &GpuBuf {
        &self
            .zeros
            .iter()
            .find(|(m, _)| *m == n)
            .expect("a zero bias exists for every projection width")
            .1
    }

    /// Consume one token at the next position and return its next-token logits row.
    pub fn feed(&mut self, token: usize) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        if self.pos >= c.ctx {
            return Err(err(format!("context of {} tokens is full", c.ctx)));
        }
        let (g, s, d) = (self.gpu, &self.s, c.dim);
        let (qd, kvd) = (c.q_dim(), c.kv_dim());
        let id = g.upload_indices(&[token], c.vocab)?;
        // The whole token is recorded into one command encoder and submitted once.
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            let mm = &self.matmul;
            g.copy_f32(&id, 0, &s.ids, 0, 1)?;
            ops_llama::gather_rows(g, &self.tok, &s.ids, &s.x, (1, d))?;
            for b in &self.blocks {
                ops_llama::rmsnorm(g, &s.x, &b.attn_norm, &s.h, (1, d), c.eps)?;
                b.wq.apply(
                    g,
                    mm,
                    (&s.h, b.bq.as_ref(), self.zero(qd)),
                    (&s.q, &s.tmp_q),
                    1,
                )?;
                b.wk.apply(
                    g,
                    mm,
                    (&s.h, b.bk.as_ref(), self.zero(kvd)),
                    (&s.k, &s.tmp_kv),
                    1,
                )?;
                b.wv.apply(
                    g,
                    mm,
                    (&s.h, b.bv.as_ref(), self.zero(kvd)),
                    (&s.v, &s.tmp_kv),
                    1,
                )?;
                let tables = (&self.cos, &self.sin);
                let (hd, rd, ty) = (c.head_dim, c.rope_dim, c.rope_type);
                ops_llama::rope(g, &s.q, tables, (1, c.heads, hd, rd), ty, self.pos)?;
                ops_llama::rope(g, &s.k, tables, (1, c.kv_heads, hd, rd), ty, self.pos)?;
                g.copy_f32(&s.k, 0, &b.kcache, self.pos * kvd, kvd)?;
                g.copy_f32(&s.v, 0, &b.vcache, self.pos * kvd, kvd)?;
                let dims = (c.heads, c.kv_heads, hd, self.pos + 1, c.ctx);
                ops_llama::attention_decode_gqa(
                    g,
                    (&s.q, &b.kcache, &b.vcache),
                    (&s.probs, &s.attn),
                    dims,
                )?;
                b.wo.apply(g, mm, (&s.attn, None, self.zero(d)), (&s.o, &s.tmp_q), 1)?;
                ops::add(g, &s.x, &s.o, &s.x2)?;
                ops_llama::rmsnorm(g, &s.x2, &b.ffn_norm, &s.h, (1, d), c.eps)?;
                b.gate.apply(
                    g,
                    mm,
                    (&s.h, None, self.zero(c.hidden)),
                    (&s.gate, &s.tmp_q),
                    1,
                )?;
                b.up.apply(
                    g,
                    mm,
                    (&s.h, None, self.zero(c.hidden)),
                    (&s.up, &s.tmp_q),
                    1,
                )?;
                ops_llama::swiglu(g, &s.gate, &s.up, &s.act)?;
                b.down
                    .apply(g, mm, (&s.act, None, self.zero(d)), (&s.f2, &s.tmp_q), 1)?;
                ops::add(g, &s.x2, &s.f2, &s.x)?;
            }
            ops_llama::rmsnorm(g, &s.x, &self.out_norm, &s.h, (1, d), c.eps)?;
            self.head.apply(
                g,
                mm,
                (&s.h, None, self.zero(c.vocab)),
                (&s.logits, &s.tmp_q),
                1,
            )?;
            Ok(())
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        self.pos += 1;
        g.read(&s.logits)
    }
}
