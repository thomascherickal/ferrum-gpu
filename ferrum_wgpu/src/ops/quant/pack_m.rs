use super::*;

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
