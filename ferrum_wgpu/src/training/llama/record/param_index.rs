use super::*;

/// Index of block `l`'s parameter `k` (0..12) in the canonical order.
pub(super) fn pi(l: usize, k: usize) -> usize {
    1 + BLOCK_PARAMS * l + k
}

pub(super) const ATTN_NORM: usize = 0;
pub(super) const WQ: usize = 1;
pub(super) const BQ: usize = 2;
pub(super) const WK: usize = 3;
pub(super) const BK: usize = 4;
pub(super) const WV: usize = 5;
pub(super) const BV: usize = 6;
pub(super) const WO: usize = 7;
pub(super) const FFN_NORM: usize = 8;
pub(super) const GATE: usize = 9;
pub(super) const UP: usize = 10;
pub(super) const DOWN: usize = 11;

/// `dx = dy * w^T` and `dw = x^T * dy` for `y = x * w` (`x` `[m,k]`, `w` `[k,n]`).
pub(super) fn lin_bwd(
    gpu: &Gpu,
    (x, w, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dw): (&GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
) -> Result<(), GpuError> {
    matmul_ex(gpu, (dy, w, dx), (m, n, k), (false, true))?;
    matmul_ex(gpu, (x, dy, dw), (k, m, n), (true, false))
}
