use super::{err, need};
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// `out[r] = tok[ids[r]] + pos[r % t]` for `rows` rows of width `c`.
pub fn embed(
    gpu: &Gpu,
    tables: (&GpuBuf, &GpuBuf, &GpuBuf),
    out: &GpuBuf,
    dims: (usize, usize, usize),
) -> Result<(), GpuError> {
    embed_at(gpu, tables, out, dims, 0)
}

/// `out[r] = tok[ids[r]] + pos[(r + pos_offset) % t]`: row `r` sits at position `pos_offset + r`
/// (decode feeds one token at its absolute position).
pub fn embed_at(
    gpu: &Gpu,
    (tok, pos, ids): (&GpuBuf, &GpuBuf, &GpuBuf),
    out: &GpuBuf,
    (rows, c, t): (usize, usize, usize),
    pos_offset: usize,
) -> Result<(), GpuError> {
    need("embed ids", ids, rows)?;
    need("embed out", out, rows * c)?;
    if t == 0 || !pos.len().is_multiple_of(c.max(1)) || pos.len() / c.max(1) < t {
        return Err(err(format!(
            "embed: positions table of {} elements cannot cover t={t}",
            pos.len()
        )));
    }
    let p = pipe!(gpu, "fwd/embed")?;
    gpu.dispatch_1d(
        &p,
        &[tok, pos, ids, out],
        &[rows as u32, c as u32, t as u32, pos_offset as u32],
        rows * c,
    )
}

/// `y = x * w + bias` for one row: `x` has `k` elements, `w` is `k x n`, `bias` and `y` have `n`.
pub fn gemv(
    gpu: &Gpu,
    (x, w, bias): (&GpuBuf, &GpuBuf, &GpuBuf),
    y: &GpuBuf,
    (k, n): (usize, usize),
) -> Result<(), GpuError> {
    need("gemv x", x, k)?;
    need("gemv w", w, k * n)?;
    need("gemv bias", bias, n)?;
    need("gemv y", y, n)?;
    let p = pipe!(gpu, "fwd/gemv")?;
    gpu.dispatch_1d(&p, &[x, w, bias, y], &[k as u32, n as u32], n)
}
