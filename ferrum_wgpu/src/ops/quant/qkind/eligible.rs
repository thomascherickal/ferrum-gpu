use super::*;

impl QKind {
    /// The GGUF tensor type, `None` for the FP8 formats (llama.cpp has none).
    pub(crate) fn ggml(self) -> Option<u32> {
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
    pub(crate) fn file_bytes(self) -> usize {
        match self {
            QKind::Q8_0 | QKind::Fp8E4M3 | QKind::Fp8E5M2 => 2 + QK,
            QKind::Q4_0 => 2 + QK / 2,
            QKind::F16 => 2 * QK,
            QKind::Fp4 => mxfp4::BLOCK,
            QKind::Q2K => ferrum_gguf::blocks::Q2_K_BLOCK,
        }
    }

    /// u32 words of one block record on the device.
    pub(crate) fn words(self) -> usize {
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
    pub(crate) fn id(self) -> u32 {
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
    pub(crate) fn is_core(self) -> bool {
        matches!(self, QKind::Q8_0 | QKind::Q4_0 | QKind::F16)
    }
}

/// Whether a matrix with this input width can be kept quantised in 32-element blocks (whole
/// blocks along `k`). Q2_K needs 256: use [`eligible_for`] when the format is known.
pub fn eligible(k: usize) -> bool {
    k > 0 && k.is_multiple_of(QK)
}
