//! Forward kernels of the Llama / Qwen stack: RMSNorm, RoPE from host-built tables, SwiGLU, an
//! embedding gather and grouped-query attention (full and KV-cached). Every function records one
//! or more dispatches on resident buffers and does not wait; read results with [`Gpu::read`].
//! The CPU oracle for each is in `ferrum_ref::llama`.

#[cfg(test)]
mod tests;

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::RopeType;

/// `y = x / sqrt(mean(x^2) + eps) * gain` per row of `x` (`rows x cols`).
pub fn rmsnorm(
    gpu: &Gpu,
    x: &GpuBuf,
    gain: &GpuBuf,
    y: &GpuBuf,
    (rows, cols): (usize, usize),
    eps: f32,
) -> Result<(), GpuError> {
    need("rmsnorm x", x, rows * cols)?;
    need("rmsnorm y", y, rows * cols)?;
    need("rmsnorm gain", gain, cols)?;
    let p = gpu.pipeline("rmsnorm", include_str!("../shaders/llama/rmsnorm.wgsl"))?;
    gpu.dispatch_1d(
        &p,
        &[x, gain, y],
        &[rows as u32, cols as u32, eps.to_bits()],
        rows,
    )
}

/// Rotate `x` (`rows x heads*hd`) in place; row `r` uses table row `table_offset + r` of the
/// `[positions, rope_dim/2]` tables built by `ferrum_ref::llama::rope_tables`. Only the first
/// `rope_dim` channels of each head rotate.
pub fn rope(
    gpu: &Gpu,
    x: &GpuBuf,
    (cos, sin): (&GpuBuf, &GpuBuf),
    (rows, heads, hd, rope_dim): (usize, usize, usize, usize),
    ty: RopeType,
    table_offset: usize,
) -> Result<(), GpuError> {
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) || rope_dim > hd {
        return Err(err(format!(
            "rope: rope_dim {rope_dim} must be even, positive and at most head_dim {hd}"
        )));
    }
    let half = rope_dim / 2;
    need("rope x", x, rows * heads * hd)?;
    need("rope sin", sin, cos.len())?;
    if cos.len() < (table_offset + rows) * half {
        return Err(err(format!(
            "rope: tables of {} elements do not cover rows {table_offset}..{}",
            cos.len(),
            table_offset + rows
        )));
    }
    let p = gpu.pipeline("rope", include_str!("../shaders/llama/rope.wgsl"))?;
    let neox = u32::from(ty == RopeType::Neox);
    gpu.dispatch_1d(
        &p,
        &[x, cos, sin],
        &[
            rows as u32,
            heads as u32,
            hd as u32,
            rope_dim as u32,
            neox,
            table_offset as u32,
        ],
        rows * heads * half,
    )
}

/// `y = silu(gate) * up`, element-wise.
pub fn swiglu(gpu: &Gpu, gate: &GpuBuf, up: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("swiglu up", up, gate.len())?;
    need("swiglu y", y, gate.len())?;
    let p = gpu.pipeline("swiglu", include_str!("../shaders/llama/swiglu.wgsl"))?;
    gpu.dispatch_1d(&p, &[gate, up, y], &[gate.len() as u32], gate.len())
}

/// `out[r] = table[ids[r]]` for `rows` rows of width `c`; `ids` come from [`Gpu::upload_indices`]
/// with the table's row count as the bound.
pub fn gather_rows(
    gpu: &Gpu,
    table: &GpuBuf,
    ids: &GpuBuf,
    out: &GpuBuf,
    (rows, c): (usize, usize),
) -> Result<(), GpuError> {
    need("gather ids", ids, rows)?;
    need("gather out", out, rows * c)?;
    if c == 0 || !table.len().is_multiple_of(c) {
        return Err(err(format!(
            "gather: table of {} elements is not whole rows of {c}",
            table.len()
        )));
    }
    let p = gpu.pipeline("gather", include_str!("../shaders/llama/gather.wgsl"))?;
    gpu.dispatch_1d(&p, &[table, ids, out], &[rows as u32, c as u32], rows * c)
}

fn check_gqa(name: &str, heads: usize, kv_heads: usize, hd: usize) -> Result<(), GpuError> {
    if heads == 0 || kv_heads == 0 || hd == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(err(format!(
            "{name}: heads {heads} must be a positive multiple of kv_heads {kv_heads} (head width {hd})"
        )));
    }
    Ok(())
}

/// Grouped-query causal attention over whole sequences. `q` and `out` are
/// `[batch*t, heads*hd]`, `k` and `v` are `[batch*t, kv_heads*hd]`, `probs` is
/// `[batch*heads*t, t]` (kept for the backward pass).
pub fn attention_gqa(
    gpu: &Gpu,
    (q, k, v): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (batch, t, heads, kv_heads, hd): (usize, usize, usize, usize, usize),
) -> Result<(), GpuError> {
    check_gqa("attention_gqa", heads, kv_heads, hd)?;
    let (qd, kvd) = (heads * hd, kv_heads * hd);
    need("attention_gqa q", q, batch * t * qd)?;
    need("attention_gqa out", out, batch * t * qd)?;
    need("attention_gqa k", k, batch * t * kvd)?;
    need("attention_gqa v", v, batch * t * kvd)?;
    need("attention_gqa probs", probs, batch * heads * t * t)?;
    let u = [
        batch as u32,
        t as u32,
        heads as u32,
        hd as u32,
        kv_heads as u32,
    ];
    let p0 = gpu.pipeline(
        "attn_gqa_scores",
        include_str!("../shaders/llama/attn_gqa_scores.wgsl"),
    )?;
    gpu.dispatch_1d(&p0, &[q, k, probs], &u, batch * heads * t * t)?;
    // The softmax only needs (batch, t, heads); it is the GPT-2 kernel, unchanged.
    let p1 = gpu.pipeline(
        "attn_softmax",
        include_str!("../shaders/fwd/attn_softmax.wgsl"),
    )?;
    gpu.dispatch_1d(&p1, &[probs], &u[..4], batch * heads * t)?;
    let p2 = gpu.pipeline(
        "attn_gqa_pv",
        include_str!("../shaders/llama/attn_gqa_pv.wgsl"),
    )?;
    gpu.dispatch_1d(&p2, &[probs, v, out], &u, batch * t * qd)
}

/// Single-query grouped-query attention over a KV cache laid out `[ctx, kv_heads*hd]`: positions
/// `0..len` are visible. `probs` is scratch of `heads * ctx`; `out` has `heads * hd` elements.
pub fn attention_decode_gqa(
    gpu: &Gpu,
    (q, kc, vc): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (heads, kv_heads, hd, len, ctx): (usize, usize, usize, usize, usize),
) -> Result<(), GpuError> {
    check_gqa("attention_decode_gqa", heads, kv_heads, hd)?;
    if len == 0 || len > ctx {
        return Err(err(format!(
            "attention_decode_gqa: len {len} must be in 1..={ctx}"
        )));
    }
    let (qd, kvd) = (heads * hd, kv_heads * hd);
    need("attention_decode_gqa q", q, qd)?;
    need("attention_decode_gqa out", out, qd)?;
    need("attention_decode_gqa kc", kc, ctx * kvd)?;
    need("attention_decode_gqa vc", vc, ctx * kvd)?;
    need("attention_decode_gqa probs", probs, heads * ctx)?;
    let u = [
        heads as u32,
        kv_heads as u32,
        hd as u32,
        len as u32,
        ctx as u32,
    ];
    let p1 = gpu.pipeline(
        "attn_gqd_probs",
        include_str!("../shaders/llama/attn_gqd_probs.wgsl"),
    )?;
    gpu.dispatch_1d(&p1, &[q, kc, probs], &u, heads)?;
    let p2 = gpu.pipeline(
        "attn_gqd_pv",
        include_str!("../shaders/llama/attn_gqd_pv.wgsl"),
    )?;
    gpu.dispatch_1d(&p2, &[probs, vc, out], &u, qd)
}
