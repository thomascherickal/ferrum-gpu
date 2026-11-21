use super::*;

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
