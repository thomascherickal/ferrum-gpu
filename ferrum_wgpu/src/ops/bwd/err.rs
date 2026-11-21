use super::*;

pub(crate) fn err(m: String) -> GpuError {
    GpuError(m)
}

pub(crate) fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
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
