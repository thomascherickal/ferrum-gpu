use super::{err, need};
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// Causal multi-head attention forward: fills `probs` (`batch*heads*t x t`) and `out` (`batch*t x heads*hd`).
pub fn attention(
    gpu: &Gpu,
    (q, k, v): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    for (name, b) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        need(&format!("attention {name}"), b, batch * t * c)?;
    }
    need("attention probs", probs, batch * heads * t * t)?;
    let u = [batch as u32, t as u32, heads as u32, hd as u32];
    // Scores are computed for every (row, key) cell in parallel, then softmaxed per row.
    let p0 = pipe!(gpu, "fwd/attn_scores")?;
    gpu.dispatch_1d(&p0, &[q, k, probs], &u, batch * heads * t * t)?;
    let p1 = pipe!(gpu, "fwd/attn_softmax")?;
    gpu.dispatch_1d(&p1, &[probs], &u, batch * heads * t)?;
    let p2 = pipe!(gpu, "fwd/attn_pv")?;
    gpu.dispatch_1d(&p2, &[probs, v, out], &u, batch * t * c)
}

/// Single-query attention over a KV cache laid out `[ctx, heads*hd]`: positions `0..len` are
/// visible. `probs` is scratch of `heads * ctx`; `out` has `heads * hd` elements.
pub fn attention_decode(
    gpu: &Gpu,
    (q, kc, vc): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (heads, hd, len, ctx): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    if len == 0 || len > ctx {
        return Err(err(format!(
            "attention_decode: len {len} must be in 1..={ctx}"
        )));
    }
    need("attention_decode q", q, c)?;
    need("attention_decode out", out, c)?;
    need("attention_decode kc", kc, ctx * c)?;
    need("attention_decode vc", vc, ctx * c)?;
    need("attention_decode probs", probs, heads * ctx)?;
    let u = [heads as u32, hd as u32, len as u32, ctx as u32];
    let p1 = pipe!(gpu, "fwd/attn_dec_probs")?;
    gpu.dispatch_1d(&p1, &[q, kc, probs], &u, heads)?;
    let p2 = pipe!(gpu, "fwd/attn_dec_pv")?;
    gpu.dispatch_1d(&p2, &[probs, vc, out], &u, c)
}
