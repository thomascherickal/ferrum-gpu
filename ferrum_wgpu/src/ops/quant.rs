//! Quantised linear layers: Q8_0 and Q4_0 weights stay packed on the device and are dequantised
//! in the kernel's inner loop, which cuts the memory traffic of memory-bound decode by 3.6x
//! (Q8_0) and 6.4x (Q4_0) against f32. The dequantised values are bit-identical to
//! `ferrum_gguf::quant::decode`; only the order of the final sum differs from the CPU.
//!
//! Device layout (our own, 4-byte aligned; the GGUF block is 34 or 18 bytes): block-major
//! records, record `b * n + col` for block `b` of output column `col`, holding the f32 bits of
//! the block's scale (converted from the file's f16 on the host, exactly) and then the quants.

#[cfg(test)]
mod tests;

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
use ferrum_gguf::f16::f16_to_f32;
use ferrum_gguf::quant::encode;
use ferrum_gguf::{GGML_Q4_0, GGML_Q8_0, QK};
use ferrum_ref::llama::transpose;

/// Block quantisation formats the kernel understands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QKind {
    /// 8-bit values with an f16 scale per 32.
    Q8_0,
    /// 4-bit values with an f16 scale per 32.
    Q4_0,
}

impl QKind {
    fn ggml(self) -> u32 {
        match self {
            QKind::Q8_0 => GGML_Q8_0,
            QKind::Q4_0 => GGML_Q4_0,
        }
    }

    /// Bytes of one block in a GGUF file.
    fn file_bytes(self) -> usize {
        match self {
            QKind::Q8_0 => 2 + QK,
            QKind::Q4_0 => 2 + QK / 2,
        }
    }

    /// u32 words of one block record on the device.
    fn words(self) -> usize {
        match self {
            QKind::Q8_0 => 1 + QK / 4,
            QKind::Q4_0 => 1 + QK / 8,
        }
    }
}

/// Whether a matrix with this input width can be kept quantised (whole blocks along `k`).
pub fn eligible(k: usize) -> bool {
    k > 0 && k.is_multiple_of(QK)
}

/// Device words a `k x n` weight occupies, or `None` if `k` is not a whole number of blocks or
/// the size overflows.
pub fn packed_words(kind: QKind, k: usize, n: usize) -> Option<usize> {
    if !eligible(k) {
        return None;
    }
    (k / QK).checked_mul(n)?.checked_mul(kind.words())
}

/// A quantised weight resident on the device: `n` output columns of `k` inputs each.
pub struct QWeight {
    buf: GpuBuf,
    kind: QKind,
    k: usize,
    n: usize,
}

impl QWeight {
    /// Block format.
    pub fn kind(&self) -> QKind {
        self.kind
    }

    /// Input width.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Output width.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Bytes of device memory the packed weight occupies.
    pub fn device_bytes(&self) -> u64 {
        (self.buf.len() * 4) as u64
    }
}

/// Repack GGUF block bytes (`n` rows of `k` elements, each row `k / 32` blocks) into the device
/// layout.
pub fn pack(kind: QKind, bytes: &[u8], k: usize, n: usize) -> Result<Vec<u32>, GpuError> {
    let words = packed_words(kind, k, n).ok_or_else(|| {
        err(format!(
            "a {k} x {n} weight is not whole {QK}-element blocks"
        ))
    })?;
    let (nb, fb, w) = (k / QK, kind.file_bytes(), kind.words());
    if bytes.len() != nb * n * fb {
        return Err(err(format!(
            "{} bytes is not the {} a {k} x {n} {kind:?} weight needs",
            bytes.len(),
            nb * n * fb
        )));
    }
    let mut out = vec![0u32; words];
    for col in 0..n {
        for b in 0..nb {
            let blk = &bytes[(col * nb + b) * fb..][..fb];
            let rec = &mut out[(b * n + col) * w..][..w];
            rec[0] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])).to_bits();
            for (dst, c) in rec[1..].iter_mut().zip(blk[2..].chunks_exact(4)) {
                *dst = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
    }
    Ok(out)
}

/// Quantise an `[in, out]` row-major f32 weight to GGUF block bytes (output columns become
/// rows, blocks run along the input).
pub fn quantise_matrix(kind: QKind, w: &[f32], k: usize, n: usize) -> Result<Vec<u8>, GpuError> {
    if !eligible(k) || w.len() != k * n {
        return Err(err(format!(
            "cannot quantise {} elements as a {k} x {n} weight in whole {QK}-blocks",
            w.len()
        )));
    }
    encode(kind.ggml(), &transpose(w, k, n)).map_err(|e| err(e.to_string()))
}

/// Upload GGUF block bytes (see [`pack`]) as a resident quantised weight.
pub fn upload_qweight(
    gpu: &Gpu,
    kind: QKind,
    bytes: &[u8],
    k: usize,
    n: usize,
) -> Result<QWeight, GpuError> {
    let words = pack(kind, bytes, k, n)?;
    let buf = gpu.upload_raw(bytemuck::cast_slice(&words), words.len())?;
    Ok(QWeight { buf, kind, k, n })
}

/// Quantise an `[in, out]` f32 weight and upload it.
pub fn upload_quantised(
    gpu: &Gpu,
    kind: QKind,
    w: &[f32],
    k: usize,
    n: usize,
) -> Result<QWeight, GpuError> {
    upload_qweight(gpu, kind, &quantise_matrix(kind, w, k, n)?, k, n)
}

/// `y[rows, n] = x[rows, k] * W`, dequantising `W` in the inner loop (no bias).
pub fn linear_q(
    gpu: &Gpu,
    x: &GpuBuf,
    w: &QWeight,
    y: &GpuBuf,
    rows: usize,
) -> Result<(), GpuError> {
    need("linear_q x", x, rows * w.k)?;
    need("linear_q y", y, rows * w.n)?;
    let kind = u32::from(w.kind == QKind::Q4_0);
    let p = pipe!(gpu, "quant/qlinear")?;
    gpu.dispatch_1d(
        &p,
        &[x, &w.buf, y],
        &[rows as u32, w.k as u32, w.n as u32, kind],
        rows * w.n,
    )
}
