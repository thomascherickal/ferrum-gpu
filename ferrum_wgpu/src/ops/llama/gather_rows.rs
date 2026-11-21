use super::*;

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
    let p = pipe!(gpu, "llama/gather")?;
    gpu.dispatch_1d(&p, &[table, ids, out], &[rows as u32, c as u32], rows * c)
}

pub(super) fn check_gqa(
    name: &str,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Result<(), GpuError> {
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
    let p0 = pipe!(gpu, "llama/attn_gqa_scores")?;
    gpu.dispatch_1d(&p0, &[q, k, probs], &u, batch * heads * t * t)?;
    // The softmax only needs (batch, t, heads); it is the GPT-2 kernel, unchanged.
    let p1 = pipe!(gpu, "fwd/attn_softmax")?;
    gpu.dispatch_1d(&p1, &[probs], &u[..4], batch * heads * t)?;
    let p2 = pipe!(gpu, "llama/attn_gqa_pv")?;
    gpu.dispatch_1d(&p2, &[probs, v, out], &u, batch * t * qd)
}
