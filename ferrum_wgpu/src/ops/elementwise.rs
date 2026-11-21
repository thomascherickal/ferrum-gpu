use super::{err, need};
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

/// `c = a + b`, element-wise.
pub fn add(gpu: &Gpu, a: &GpuBuf, b: &GpuBuf, c: &GpuBuf) -> Result<(), GpuError> {
    let n = a.len();
    need("add b", b, n)?;
    need("add c", c, n)?;
    let p = pipe!(gpu, "fwd/add")?;
    gpu.dispatch_1d(&p, &[a, b, c], &[n as u32], n)
}

/// `y[i] = x[i] + bias[i % bias.len()]`.
pub fn add_bias(gpu: &Gpu, x: &GpuBuf, bias: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("add_bias y", y, x.len())?;
    if bias.is_empty() || !x.len().is_multiple_of(bias.len()) {
        return Err(err(format!(
            "add_bias: {} elements is not a multiple of bias length {}",
            x.len(),
            bias.len()
        )));
    }
    let p = pipe!(gpu, "fwd/bias_add")?;
    gpu.dispatch_1d(
        &p,
        &[x, bias, y],
        &[x.len() as u32, bias.len() as u32],
        x.len(),
    )
}

/// `y = gelu(x)` (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("gelu y", y, x.len())?;
    let p = pipe!(gpu, "fwd/gelu")?;
    gpu.dispatch_1d(&p, &[x, y], &[x.len() as u32], x.len())
}

/// Row-wise LayerNorm of `x` (`rows x cols`) into `y`.
pub fn layernorm(
    gpu: &Gpu,
    x: &GpuBuf,
    gain: &GpuBuf,
    shift: &GpuBuf,
    y: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    need("layernorm x", x, rows * cols)?;
    need("layernorm y", y, rows * cols)?;
    need("layernorm gain", gain, cols)?;
    need("layernorm shift", shift, cols)?;
    let p = pipe!(gpu, "fwd/layernorm")?;
    gpu.dispatch_1d(&p, &[x, gain, shift, y], &[rows as u32, cols as u32], rows)
}
