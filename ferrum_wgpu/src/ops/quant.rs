//! Quantised linear layers: Q8_0 and Q4_0 weights stay packed on the device and are dequantised
//! in the kernel's inner loop, which cuts the memory traffic of memory-bound decode by 3.6x
//! (Q8_0) and 6.4x (Q4_0) against f32. The dequantised values are bit-identical to
//! `ferrum_gguf::quant::decode`; only the order of the final sum differs from the CPU.
//!
//! Device layout (our own, 4-byte aligned; the GGUF block is 34 or 18 bytes): block-major
//! records, record `b * n + col` for block `b` of output column `col`, holding the f32 bits of
//! the block's scale (converted from the file's f16 on the host, exactly) and then the quants.

#[cfg(test)]
mod ext_tests;
#[cfg(test)]
mod tests;

use crate::device::{Gpu, GpuError};
use crate::ops::{err, need};
use crate::tensor::GpuBuf;
use ferrum_gguf::f16::f16_to_f32;
use ferrum_gguf::fp8::{self, Fp8};
use ferrum_gguf::mxfp4;
use ferrum_gguf::quant::{decode, encode};
use ferrum_gguf::{GGML_F16, GGML_MXFP4, GGML_Q2_K, GGML_Q4_0, GGML_Q8_0, QK, QK_K};
use ferrum_ref::llama::transpose;

/// Block quantisation formats the kernel understands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QKind {
    /// 8-bit values with an f16 scale per 32.
    Q8_0,
    /// 4-bit values with an f16 scale per 32.
    Q4_0,
    /// Plain f16 values, two per word, no scale (half the memory of f32; the kernel unpacks with
    /// `unpack2x16float`, which every WGSL implementation has, so it needs no `shader-f16`
    /// feature). Compute stays f32.
    F16,
    /// MXFP4: 4-bit E2M1 values with a power-of-two scale per 32 (ggml type 39).
    Fp4,
    /// 8-bit E4M3 floats with an f16 scale per 32. Not a GGUF type: made when a model is loaded.
    Fp8E4M3,
    /// 8-bit E5M2 floats with an f16 scale per 32. Not a GGUF type: made when a model is loaded.
    Fp8E5M2,
    /// Q2_K: 2-bit values in super-blocks of 256 with 4-bit sub-block scales and mins.
    Q2K,
}

impl QKind {
    /// The GGUF tensor type, `None` for the FP8 formats (llama.cpp has none).
    fn ggml(self) -> Option<u32> {
        Some(match self {
            QKind::Q8_0 => GGML_Q8_0,
            QKind::Q4_0 => GGML_Q4_0,
            QKind::F16 => GGML_F16,
            QKind::Fp4 => GGML_MXFP4,
            QKind::Q2K => GGML_Q2_K,
            QKind::Fp8E4M3 | QKind::Fp8E5M2 => return None,
        })
    }

    /// Elements per block along the input axis.
    pub fn elems(self) -> usize {
        if self == QKind::Q2K {
            QK_K
        } else {
            QK
        }
    }

    /// Bytes of one block in a GGUF file (the FP8 formats use Q8_0's layout).
    fn file_bytes(self) -> usize {
        match self {
            QKind::Q8_0 | QKind::Fp8E4M3 | QKind::Fp8E5M2 => 2 + QK,
            QKind::Q4_0 => 2 + QK / 2,
            QKind::F16 => 2 * QK,
            QKind::Fp4 => mxfp4::BLOCK,
            QKind::Q2K => ferrum_gguf::blocks::Q2_K_BLOCK,
        }
    }

    /// u32 words of one block record on the device.
    fn words(self) -> usize {
        match self {
            QKind::Q8_0 => 1 + QK / 4,
            QKind::Q4_0 => 1 + QK / 8,
            QKind::F16 => QK / 2,
            QKind::Fp4 => 1 + QK / 8,
            QKind::Fp8E4M3 | QKind::Fp8E5M2 => 1 + QK / 4,
            QKind::Q2K => 2 + 16 / 4 + QK_K / 16,
        }
    }

    /// The kind number the kernels switch on.
    fn id(self) -> u32 {
        match self {
            QKind::Q8_0 => 0,
            QKind::Q4_0 => 1,
            QKind::F16 => 2,
            QKind::Fp4 => 3,
            QKind::Fp8E4M3 => 4,
            QKind::Fp8E5M2 => 5,
            QKind::Q2K => 6,
        }
    }

    /// Does the kernel set of the first three formats handle this one (else `qlinear_ext` does)?
    fn is_core(self) -> bool {
        matches!(self, QKind::Q8_0 | QKind::Q4_0 | QKind::F16)
    }
}

/// Whether a matrix with this input width can be kept quantised in 32-element blocks (whole
/// blocks along `k`). Q2_K needs 256: use [`eligible_for`] when the format is known.
pub fn eligible(k: usize) -> bool {
    k > 0 && k.is_multiple_of(QK)
}

/// Whether a matrix with input width `k` fills whole blocks of `kind`.
pub fn eligible_for(kind: QKind, k: usize) -> bool {
    k > 0 && k.is_multiple_of(kind.elems())
}

/// The format a matrix of input width `k` is actually stored in when `kind` is requested: Q2_K
/// needs 256-wide blocks, so a width that is a whole number of 32s uses Q4_0 instead (llama.cpp's own
/// fallback, and what the file writer does); everything else is `kind`.
pub fn effective(kind: QKind, k: usize) -> QKind {
    if kind == QKind::Q2K && !eligible_for(kind, k) && eligible(k) {
        QKind::Q4_0
    } else {
        kind
    }
}

/// Device words a `k x n` weight occupies, or `None` if `k` is not a whole number of blocks or
/// the size overflows.
pub fn packed_words(kind: QKind, k: usize, n: usize) -> Option<usize> {
    if !eligible_for(kind, k) {
        return None;
    }
    (k / kind.elems()).checked_mul(n)?.checked_mul(kind.words())
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
            "a {k} x {n} weight is not whole {}-element blocks",
            kind.elems()
        ))
    })?;
    let (nb, fb, w) = (k / kind.elems(), kind.file_bytes(), kind.words());
    if bytes.len() != nb * n * fb {
        return Err(err(format!(
            "{} bytes is not the {} a {k} x {n} {kind:?} weight needs",
            bytes.len(),
            nb * n * fb
        )));
    }
    let mut out = vec![0u32; words];
    let le = |dst: &mut [u32], src: &[u8]| {
        for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
            *d = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    };
    let half = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]])).to_bits();
    for col in 0..n {
        for b in 0..nb {
            let blk = &bytes[(col * nb + b) * fb..][..fb];
            let rec = &mut out[(b * n + col) * w..][..w];
            match kind {
                // no scale: the whole record is values
                QKind::F16 => le(rec, blk),
                QKind::Q8_0 | QKind::Q4_0 | QKind::Fp8E4M3 | QKind::Fp8E5M2 => {
                    rec[0] = half(blk);
                    le(&mut rec[1..], &blk[2..]);
                }
                QKind::Fp4 => {
                    rec[0] = mxfp4::half_scale(blk[0]).to_bits();
                    le(&mut rec[1..], &blk[1..]);
                }
                // scales[16], qs[64], d, dmin in the file; d, dmin, scales, qs on the device
                QKind::Q2K => {
                    rec[0] = half(&blk[80..]);
                    rec[1] = half(&blk[82..]);
                    le(&mut rec[2..6], &blk[..16]);
                    le(&mut rec[6..], &blk[16..80]);
                }
            }
        }
    }
    Ok(out)
}

/// Quantise an `[in, out]` row-major f32 weight to GGUF block bytes (output columns become
/// rows, blocks run along the input).
pub fn quantise_matrix(kind: QKind, w: &[f32], k: usize, n: usize) -> Result<Vec<u8>, GpuError> {
    if !eligible_for(kind, k) || w.len() != k * n {
        return Err(err(format!(
            "cannot quantise {} elements as a {k} x {n} weight in whole {}-blocks",
            w.len(),
            kind.elems()
        )));
    }
    let rows = transpose(w, k, n);
    match kind {
        QKind::Fp8E4M3 => Ok(fp8::encode(Fp8::E4M3, &rows)),
        QKind::Fp8E5M2 => Ok(fp8::encode(Fp8::E5M2, &rows)),
        other => encode(other.ggml().expect("a GGUF type"), &rows).map_err(|e| err(e.to_string())),
    }
}

/// The f32 values `bytes` (block bytes from [`quantise_matrix`], `n` elements) decode to: what the
/// kernel computes with, formed on the CPU.
pub fn dequantise_bytes(kind: QKind, bytes: &[u8], n: usize) -> Result<Vec<f32>, GpuError> {
    match kind {
        QKind::Fp8E4M3 => Ok(fp8::decode(Fp8::E4M3, bytes)),
        QKind::Fp8E5M2 => Ok(fp8::decode(Fp8::E5M2, bytes)),
        other => {
            decode(other.ggml().expect("a GGUF type"), bytes, n).map_err(|e| err(e.to_string()))
        }
    }
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

/// Which kernel runs a quantised linear layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPath {
    /// One invocation per output element (the first kernel, kept for the cases where the others
    /// measured slower, and as the baseline).
    Naive,
    /// One workgroup per output column, for a single row (decode).
    Gemv,
    /// A 16 x 16 tile with the weight block shared across 16 rows (prefill).
    Tiled,
    /// The kernel [`choose`] picks.
    Auto,
}

/// The kernel to use for `rows` rows of a `k` x `n` weight, from measurements on two adapters
/// (an Intel Xe and an NVIDIA MX330, `cargo test --release -p ferrum_wgpu --lib -- --ignored
/// bench_quant`, recorded in `docs/gui-studio-plan.md`): the tiled kernel beat the naive one for
/// Q8_0 and F16 on both and was slower or level for Q4_0; the per-column kernel beat it when the
/// output is no wider than the input and lost on the MX330 when it is wider.
// ponytail: two adapters and two shapes of rule; re-measure on a new adapter with the bench above
// before trusting it there (an Apple GPU has not been run).
pub fn choose(rows: usize, kind: QKind, k: usize, n: usize) -> QPath {
    match (rows, kind) {
        (_, k) if !k.is_core() => QPath::Naive,
        (1, _) if n <= k => QPath::Gemv,
        (1, _) | (_, QKind::Q4_0) => QPath::Naive,
        _ => QPath::Tiled,
    }
}

/// `y[rows, n] = x[rows, k] * W`, dequantising `W` in the inner loop (no bias).
pub fn linear_q(
    gpu: &Gpu,
    x: &GpuBuf,
    w: &QWeight,
    y: &GpuBuf,
    rows: usize,
) -> Result<(), GpuError> {
    linear_q_with(gpu, x, w, y, rows, QPath::Auto)
}

/// [`linear_q`] on a chosen kernel (`Gemv` takes exactly one row).
pub fn linear_q_with(
    gpu: &Gpu,
    x: &GpuBuf,
    w: &QWeight,
    y: &GpuBuf,
    rows: usize,
    path: QPath,
) -> Result<(), GpuError> {
    need("linear_q x", x, rows * w.k)?;
    need("linear_q y", y, rows * w.n)?;
    let uniform = [rows as u32, w.k as u32, w.n as u32, w.kind.id()];
    let buffers = [x, &w.buf, y];
    let path = if path == QPath::Auto {
        choose(rows, w.kind, w.k, w.n)
    } else {
        path
    };
    // ponytail: the formats added after Q8_0/Q4_0/F16 have one kernel (one invocation per output
    // element); a per-column and a tiled variant can be added and chosen from measurements.
    if !w.kind.is_core() {
        let p = pipe!(gpu, "quant/qlinear_ext")?;
        return gpu.dispatch_1d(&p, &buffers, &uniform, rows * w.n);
    }
    match path {
        QPath::Naive | QPath::Auto => {
            let p = pipe!(gpu, "quant/qlinear")?;
            gpu.dispatch_1d(&p, &buffers, &uniform, rows * w.n)
        }
        QPath::Gemv => {
            if rows != 1 {
                return Err(err(format!(
                    "the per-column kernel takes one row, not {rows}"
                )));
            }
            let p = pipe!(gpu, "quant/qgemv")?;
            let x_groups = w.n.clamp(1, 65535) as u32;
            let y_groups = w.n.div_ceil(65535).max(1) as u32;
            gpu.dispatch(&p, &buffers, &uniform, (x_groups, y_groups))
        }
        QPath::Tiled => {
            let p = pipe!(gpu, "quant/qlinear_tiled")?;
            let groups = crate::runtime::groups_for((w.n, rows), (16, 16), gpu.max_workgroups())?;
            gpu.dispatch(&p, &buffers, &uniform, groups)
        }
    }
}
