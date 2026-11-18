//! One linear layer on the device: f32 or packed-quantised weights, applied to one row (decode) or
//! many (prefill, scoring), with an optional bias.

use super::shard::{blocks, column_block};
use crate::device::{Gpu, GpuError};
use crate::matmul::Matmul;
use crate::ops;
use crate::ops_quant::{effective, linear_q, packed_words, upload_quantised, QKind, QWeight};
use crate::tensor::GpuBuf;

/// A weight matrix `[k, n]` resident on the device.
pub(super) enum Lin {
    /// Plain f32.
    F32 { w: GpuBuf, k: usize, n: usize },
    /// Block-quantised, dequantised in the kernel.
    Q(QWeight),
    /// Larger than one binding: column blocks, each an ordinary `Lin`, with a zero bias of its own
    /// width for the single-row kernel. Each block's output is copied to its columns of `y`.
    // ponytail: scratch for the block outputs is allocated on every call; cache it if a sharded
    // model is ever decoded fast enough for the allocations to show up.
    Sharded {
        parts: Vec<(usize, usize, Lin, GpuBuf)>,
        n: usize,
    },
}

impl Lin {
    /// Upload `w` (`[k, n]` row-major), quantised to `quant` when that is requested and `k` is a
    /// whole number of blocks; otherwise f32.
    pub(super) fn upload(
        gpu: &Gpu,
        w: &[f32],
        (k, n): (usize, usize),
        quant: Option<QKind>,
        limit: u64,
    ) -> Result<Lin, GpuError> {
        let quant = quant.map(|q| effective(q, k));
        let col_bytes = match quant.filter(|&q| packed_words(q, k, n).is_some()) {
            Some(q) => packed_words(q, k, 1).unwrap_or(k) as u64 * 4,
            None => k as u64 * 4,
        };
        let cols = blocks(n, col_bytes, limit)?;
        if cols.len() > 1 {
            let parts = cols
                .into_iter()
                .map(|(c0, width)| {
                    let sub = column_block(w, (k, n), (c0, width));
                    let lin = Lin::upload(gpu, &sub, (k, width), quant, limit)?;
                    Ok((c0, width, lin, gpu.zeros(width)?))
                })
                .collect::<Result<Vec<_>, GpuError>>()?;
            return Ok(Lin::Sharded { parts, n });
        }
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
            Lin::Sharded { parts, n } => {
                let assembled = bias.map(|_| gpu.zeros(rows * n)).transpose()?;
                let dst = assembled.as_ref().unwrap_or(y);
                for (c0, width, lin, zero) in parts {
                    let (out, scratch) = (gpu.zeros(rows * width)?, gpu.zeros(rows * width)?);
                    lin.apply(gpu, mm, (x, None, zero), (&out, &scratch), rows)?;
                    for r in 0..rows {
                        gpu.copy_f32(&out, r * width, dst, r * n + c0, *width)?;
                    }
                }
                match (assembled.as_ref(), bias) {
                    (Some(a), Some(b)) => ops::add_bias(gpu, a, b, y),
                    _ => Ok(()),
                }
            }
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
            Lin::Sharded { parts, .. } => parts.iter().map(|p| p.2.device_bytes()).sum(),
        }
    }
}
