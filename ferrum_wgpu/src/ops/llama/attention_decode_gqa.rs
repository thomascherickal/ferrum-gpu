use super::*;

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
    let p1 = pipe!(gpu, "llama/attn_gqd_probs")?;
    gpu.dispatch_1d(&p1, &[q, kc, probs], &u, heads)?;
    let p2 = pipe!(gpu, "llama/attn_gqd_pv")?;
    gpu.dispatch_1d(&p2, &[probs, vc, out], &u, qd)
}
