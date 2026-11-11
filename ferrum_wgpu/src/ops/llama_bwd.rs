//! Backward kernels of the Llama / Qwen stack: RMSNorm, SwiGLU and grouped-query attention.
//! (RoPE's backward is [`crate::ops_llama::rope`] called with a negated sine table: the
//! transpose of a rotation is the rotation by the opposite angle. Linear layers reuse
//! [`crate::ops_bwd::matmul_ex`].) Same conventions as [`crate::ops_bwd`]: resident buffers,
//! recorded without waiting, every reduction in a fixed ascending order, no atomics. The CPU
//! oracle for each is in `ferrum_ref::llama`.

#[cfg(test)]
mod tests;

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;

/// RMSNorm backward. `stats` is scratch of `rows` (the per-row inverse RMS).
pub fn rmsnorm_bwd(
    gpu: &Gpu,
    (x, gain, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dgain): (&GpuBuf, &GpuBuf),
    stats: &GpuBuf,
    (rows, cols): (usize, usize),
    eps: f32,
) -> Result<(), GpuError> {
    for (n, b) in [("x", x), ("dy", dy), ("dx", dx)] {
        need(&format!("rmsnorm_bwd {n}"), b, rows * cols)?;
    }
    need("rmsnorm_bwd gain", gain, cols)?;
    need("rmsnorm_bwd dgain", dgain, cols)?;
    need("rmsnorm_bwd stats", stats, rows)?;
    let p1 = gpu.pipeline(
        "rms_bwd_dx",
        include_str!("../shaders/llama/rms_bwd_dx.wgsl"),
    )?;
    gpu.dispatch_1d(
        &p1,
        &[x, gain, dy, dx, stats],
        &[rows as u32, cols as u32, eps.to_bits()],
        rows,
    )?;
    let p2 = gpu.pipeline(
        "rms_bwd_gain",
        include_str!("../shaders/llama/rms_bwd_gain.wgsl"),
    )?;
    gpu.dispatch_1d(
        &p2,
        &[x, dy, stats, dgain],
        &[rows as u32, cols as u32],
        cols,
    )
}

/// SwiGLU backward: `dgate` and `dup` from `gate`, `up` and `dact`.
pub fn swiglu_bwd(
    gpu: &Gpu,
    (gate, up, dact): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dgate, dup): (&GpuBuf, &GpuBuf),
) -> Result<(), GpuError> {
    for (n, b) in [("up", up), ("dact", dact), ("dgate", dgate), ("dup", dup)] {
        need(&format!("swiglu_bwd {n}"), b, gate.len())?;
    }
    let p = gpu.pipeline(
        "swiglu_bwd",
        include_str!("../shaders/llama/swiglu_bwd.wgsl"),
    )?;
    gpu.dispatch_1d(
        &p,
        &[gate, up, dact, dgate, dup],
        &[gate.len() as u32],
        gate.len(),
    )
}

/// Grouped-query causal attention backward from the stored probabilities. `q`/`dq`/`dout` are
/// `[batch*t, heads*hd]`, `k`/`v`/`dk`/`dv` are `[batch*t, kv_heads*hd]`, `probs` is
/// `[batch*heads*t, t]`. `dot` (`batch*heads*t`) and `ds` (`batch*heads*t*t`) are scratch. Query
/// heads that share a KV head all contribute to its `dk` and `dv`.
pub fn attention_gqa_bwd(
    gpu: &Gpu,
    (q, k, v, probs, dout): (&GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf),
    (dq, dk, dv): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dot, ds): (&GpuBuf, &GpuBuf),
    (batch, t, heads, kv_heads, hd): (usize, usize, usize, usize, usize),
) -> Result<(), GpuError> {
    if heads == 0 || kv_heads == 0 || hd == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(err(format!(
            "attention_gqa_bwd: heads {heads} must be a positive multiple of kv_heads {kv_heads}"
        )));
    }
    let (qd, kvd) = (heads * hd, kv_heads * hd);
    for (n, b) in [("q", q), ("dq", dq), ("dout", dout)] {
        need(&format!("attention_gqa_bwd {n}"), b, batch * t * qd)?;
    }
    for (n, b) in [("k", k), ("v", v), ("dk", dk), ("dv", dv)] {
        need(&format!("attention_gqa_bwd {n}"), b, batch * t * kvd)?;
    }
    need("attention_gqa_bwd probs", probs, batch * heads * t * t)?;
    need("attention_gqa_bwd dot", dot, batch * heads * t)?;
    need("attention_gqa_bwd ds", ds, batch * heads * t * t)?;
    let u = [
        batch as u32,
        t as u32,
        heads as u32,
        hd as u32,
        kv_heads as u32,
    ];
    let (rows, cells) = (batch * heads * t, batch * heads * t * t);
    // dP = dO V^T per cell, then dS = P (dP - rowdot) / sqrt(hd) in place (the GPT-2 kernels,
    // which only see the [batch*heads*t, t] layout), then dQ, dK, dV as gathers over dS and P.
    let p = gpu.pipeline(
        "attn_gqa_bwd_dp",
        include_str!("../shaders/llama/attn_gqa_bwd_dp.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[dout, v, ds], &u, cells)?;
    let p = gpu.pipeline(
        "attn_bwd_dot",
        include_str!("../shaders/bwd/attn_bwd_dot.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[probs, ds, dot], &u[..4], rows)?;
    let p = gpu.pipeline(
        "attn_bwd_ds",
        include_str!("../shaders/bwd/attn_bwd_ds.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[probs, dot, ds], &u[..4], cells)?;
    let p = gpu.pipeline(
        "attn_gqa_bwd_dv",
        include_str!("../shaders/llama/attn_gqa_bwd_dv.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[probs, dout, dv], &u, batch * t * kvd)?;
    let p = gpu.pipeline(
        "attn_gqa_bwd_dq",
        include_str!("../shaders/llama/attn_gqa_bwd_dq.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[ds, k, dq], &u, batch * t * qd)?;
    let p = gpu.pipeline(
        "attn_gqa_bwd_dk",
        include_str!("../shaders/llama/attn_gqa_bwd_dk.wgsl"),
    )?;
    gpu.dispatch_1d(&p, &[ds, q, dk], &u, batch * t * kvd)
}
