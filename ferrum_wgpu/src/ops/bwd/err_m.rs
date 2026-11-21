use super::*;

pub(super) fn err(m: String) -> GpuError {
    GpuError(m)
}

pub(super) fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(err(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

/// `C[m,n] = op(A) * op(B)` where `a_t` means A is stored `[k, m]` and `b_t` means B is stored `[n, k]`.
pub fn matmul_ex(
    gpu: &Gpu,
    (a, b, c): (&GpuBuf, &GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
    (a_t, b_t): (bool, bool),
) -> Result<(), GpuError> {
    need("matmul_ex a", a, m * k)?;
    need("matmul_ex b", b, k * n)?;
    need("matmul_ex c", c, m * n)?;
    let p = pipe!(gpu, "matmul/matmul_blocked4t")?;
    let groups = groups_for((n, m), (64, 64), gpu.max_workgroups())?;
    let flags = u32::from(a_t) | (u32::from(b_t) << 1);
    gpu.dispatch(
        &p,
        &[a, b, c],
        &[m as u32, k as u32, n as u32, flags],
        groups,
    )
}

/// `out[c] = sum_r x[r, c]` for `x` of `rows x cols`.
pub fn colsum(
    gpu: &Gpu,
    x: &GpuBuf,
    out: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    need("colsum x", x, rows * cols)?;
    need("colsum out", out, cols)?;
    let p = pipe!(gpu, "bwd/colsum")?;
    gpu.dispatch_1d(&p, &[x, out], &[rows as u32, cols as u32], cols)
}

/// Gradient of `y = x w + b` (`x` `[m,k]`, `w` `[k,n]`): fills `dx [m,k]`, `dw [k,n]`, `db [n]`.
pub fn linear_bwd(
    gpu: &Gpu,
    (x, w, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dw, db): (&GpuBuf, &GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
) -> Result<(), GpuError> {
    matmul_ex(gpu, (dy, w, dx), (m, n, k), (false, true))?; // dx = dy * w^T
    matmul_ex(gpu, (x, dy, dw), (k, m, n), (true, false))?; // dw = x^T * dy
    colsum(gpu, dy, db, (m, n))
}

/// `df = dact * gelu'(f1)`.
pub fn gelu_bwd(gpu: &Gpu, dact: &GpuBuf, f1: &GpuBuf, df: &GpuBuf) -> Result<(), GpuError> {
    need("gelu_bwd f1", f1, dact.len())?;
    need("gelu_bwd df", df, dact.len())?;
    let p = pipe!(gpu, "bwd/gelu_bwd")?;
    gpu.dispatch_1d(&p, &[dact, f1, df], &[dact.len() as u32], dact.len())
}

/// LayerNorm backward. `stats` is scratch of `2 * rows` (per-row mean and inverse std).
pub fn layernorm_bwd(
    gpu: &Gpu,
    (x, gain, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dgain, dshift): (&GpuBuf, &GpuBuf, &GpuBuf),
    stats: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    for (n, b) in [("x", x), ("dy", dy), ("dx", dx)] {
        need(&format!("layernorm_bwd {n}"), b, rows * cols)?;
    }
    for (n, b) in [("gain", gain), ("dgain", dgain), ("dshift", dshift)] {
        need(&format!("layernorm_bwd {n}"), b, cols)?;
    }
    need("layernorm_bwd stats", stats, 2 * rows)?;
    let u = [rows as u32, cols as u32];
    let p1 = pipe!(gpu, "bwd/ln_bwd_dx")?;
    gpu.dispatch_1d(&p1, &[x, gain, dy, dx, stats], &u, rows)?;
    let p2 = pipe!(gpu, "bwd/ln_bwd_params")?;
    gpu.dispatch_1d(&p2, &[x, dy, stats, dgain, dshift], &u, cols)
}

/// Causal attention backward from the stored probabilities. `dot` (`batch*heads*t`) and `ds`
/// (`batch*heads*t*t`) are scratch.
pub fn attention_bwd(
    gpu: &Gpu,
    (q, k, v, probs, dout): (&GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf),
    (dq, dk, dv): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dot, ds): (&GpuBuf, &GpuBuf),
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    for (n, b) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("dout", dout),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    ] {
        need(&format!("attention_bwd {n}"), b, batch * t * c)?;
    }
    need("attention_bwd probs", probs, batch * heads * t * t)?;
    need("attention_bwd dot", dot, batch * heads * t)?;
    need("attention_bwd ds", ds, batch * heads * t * t)?;
    let u = [batch as u32, t as u32, heads as u32, hd as u32];
    let (rows, cells, outs) = (batch * heads * t, batch * heads * t * t, batch * t * c);
    // dP = dO V^T once per cell, then dS = P (dP - rowdot) / sqrt(hd) in place; dQ, dK and dV
    // are plain gathers over dS / P. (Recomputing dP per output element was 64x redundant.)
    let p = pipe!(gpu, "bwd/attn_bwd_dp")?;
    gpu.dispatch_1d(&p, &[dout, v, ds], &u, cells)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dot")?;
    gpu.dispatch_1d(&p, &[probs, ds, dot], &u, rows)?;
    let p = pipe!(gpu, "bwd/attn_bwd_ds")?;
    gpu.dispatch_1d(&p, &[probs, dot, ds], &u, cells)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dv")?;
    gpu.dispatch_1d(&p, &[probs, dout, dv], &u, outs)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dq")?;
    gpu.dispatch_1d(&p, &[ds, k, dq], &u, outs)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dk")?;
    gpu.dispatch_1d(&p, &[ds, q, dk], &u, outs)
}
