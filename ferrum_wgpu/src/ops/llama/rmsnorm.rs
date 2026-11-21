use super::*;

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
    let p = pipe!(gpu, "llama/rmsnorm")?;
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
    let p = pipe!(gpu, "llama/rope")?;
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
    let p = pipe!(gpu, "llama/swiglu")?;
    gpu.dispatch_1d(&p, &[gate, up, y], &[gate.len() as u32], gate.len())
}
