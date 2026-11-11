//! One linear layer on the device: f32 or packed-quantised weights, applied to one row (decode) or
//! many (prefill, scoring), with an optional bias.

use crate::device::{Gpu, GpuError};
use crate::matmul::Matmul;
use crate::ops;
use crate::ops_quant::{linear_q, packed_words, upload_quantised, QKind, QWeight};
use crate::tensor::GpuBuf;

/// A weight matrix `[k, n]` resident on the device.
pub(super) enum Lin {
    /// Plain f32.
    F32 { w: GpuBuf, k: usize, n: usize },
    /// Block-quantised, dequantised in the kernel.
    Q(QWeight),
}

impl Lin {
    /// Upload `w` (`[k, n]` row-major), quantised to `quant` when that is requested and `k` is a
    /// whole number of blocks; otherwise f32.
    pub(super) fn upload(
        gpu: &Gpu,
        w: &[f32],
        (k, n): (usize, usize),
        quant: Option<QKind>,
    ) -> Result<Lin, GpuError> {
        match quant.filter(|&q| packed_words(q, k, n).is_some()) {
            Some(kind) => Ok(Lin::Q(upload_quantised(gpu, kind, w, k, n)?)),
            None => Ok(Lin::F32 {
                w: gpu.upload(w)?,
                k,
                n,
            }),
        }
    }

    /// `y = x * W (+ bias)` for `rows` rows. `tmp` is scratch of `rows * n` elements, used only
    /// when a bias must be added to a product computed elsewhere. `zero` is a zero bias of `n`
    /// elements for the single-row f32 kernel, which always adds one.
    pub(super) fn apply(
        &self,
        gpu: &Gpu,
        mm: &Matmul,
        (x, bias, zero): (&GpuBuf, Option<&GpuBuf>, &GpuBuf),
        (y, tmp): (&GpuBuf, &GpuBuf),
        rows: usize,
    ) -> Result<(), GpuError> {
        match self {
            Lin::F32 { w, k, n } if rows == 1 => {
                ops::gemv(gpu, (x, w, bias.unwrap_or(zero)), y, (*k, *n))
            }
            Lin::F32 { w, k, n } => match bias {
                Some(b) => {
                    mm.submit(gpu, x, w, tmp, (rows, *k, *n))?;
                    ops::add_bias(gpu, tmp, b, y)
                }
                None => mm.submit(gpu, x, w, y, (rows, *k, *n)),
            },
            Lin::Q(q) => match bias {
                Some(b) => {
                    linear_q(gpu, x, q, tmp, rows)?;
                    ops::add_bias(gpu, tmp, b, y)
                }
                None => linear_q(gpu, x, q, y, rows),
            },
        }
    }

    /// Bytes of device memory this weight occupies.
    #[cfg(test)]
    pub(super) fn device_bytes(&self) -> u64 {
        match self {
            Lin::F32 { w, .. } => (w.len() * 4) as u64,
            Lin::Q(q) => q.device_bytes(),
        }
    }
}
