use super::*;

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
    let p = pipe!(gpu, "llama/attn_gqa_bwd_dp")?;
    gpu.dispatch_1d(&p, &[dout, v, ds], &u, cells)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dot")?;
    gpu.dispatch_1d(&p, &[probs, ds, dot], &u[..4], rows)?;
    let p = pipe!(gpu, "bwd/attn_bwd_ds")?;
    gpu.dispatch_1d(&p, &[probs, dot, ds], &u[..4], cells)?;
    let p = pipe!(gpu, "llama/attn_gqa_bwd_dv")?;
    gpu.dispatch_1d(&p, &[probs, dout, dv], &u, batch * t * kvd)?;
    let p = pipe!(gpu, "llama/attn_gqa_bwd_dq")?;
    gpu.dispatch_1d(&p, &[ds, k, dq], &u, batch * t * qd)?;
    let p = pipe!(gpu, "llama/attn_gqa_bwd_dk")?;
    gpu.dispatch_1d(&p, &[ds, q, dk], &u, batch * t * kvd)
}
