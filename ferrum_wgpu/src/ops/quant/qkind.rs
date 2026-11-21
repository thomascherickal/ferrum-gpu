use super::*;

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
    pub(super) fn ggml(self) -> Option<u32> {
        Some(match self {
            QKind::Q8_0 => GGML_Q8_0,
            QKind::Q4_0 => GGML_Q4_0,
            QKind::F16 => GGML_F16,
            QKind::Fp4 => GGML_MXFP4,
            QKind::Q2K => GGML_Q2_K,
            QKind::Fp8E4M3 | QKind::Fp8E5M2 => return None,
        })
    }

    /// The name reports and tables print.
    pub fn label(self) -> &'static str {
        match self {
            QKind::Q8_0 => "Q8_0",
            QKind::Q4_0 => "Q4_0",
            QKind::F16 => "F16",
            QKind::Fp4 => "MXFP4",
            QKind::Fp8E4M3 => "FP8 E4M3",
            QKind::Fp8E5M2 => "FP8 E5M2",
            QKind::Q2K => "Q2_K",
        }
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
    pub(super) fn file_bytes(self) -> usize {
        match self {
            QKind::Q8_0 | QKind::Fp8E4M3 | QKind::Fp8E5M2 => 2 + QK,
            QKind::Q4_0 => 2 + QK / 2,
            QKind::F16 => 2 * QK,
            QKind::Fp4 => mxfp4::BLOCK,
            QKind::Q2K => ferrum_gguf::blocks::Q2_K_BLOCK,
        }
    }

    /// u32 words of one block record on the device.
    pub(super) fn words(self) -> usize {
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
    pub(super) fn id(self) -> u32 {
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
    pub(super) fn is_core(self) -> bool {
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
    pub(super) buf: GpuBuf,
    pub(super) kind: QKind,
    pub(super) k: usize,
    pub(super) n: usize,
}
