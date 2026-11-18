//! The precision a model is written in, and which GGUF tensor type a matrix gets under it.
//! One place, shared by the `gpt2` and `llama` writers (vectors, biases and norms stay f32).

use ferrum_gguf::{GGML_F16, GGML_F32, GGML_MXFP4, GGML_Q2_K, GGML_Q4_0, GGML_Q8_0, QK, QK_K};

/// On-disk precision for matrices (vectors are always F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// 32-bit float.
    F32,
    /// 16-bit float (IEEE half), any input width.
    F16,
    /// Q8_0 where the input dimension is a multiple of 32, else F32.
    Q8_0,
    /// Q4_0 where the input dimension is a multiple of 32, else F32.
    Q4_0,
    /// MXFP4 (4-bit E2M1 values, a power-of-two scale per 32) where the input dimension is a
    /// multiple of 32, else F32. llama.cpp reads it; ferrum-cpu does not.
    Fp4,
    /// Q2_K (2-bit, super-blocks of 256) where the input dimension is a multiple of 256, else
    /// Q4_0 where it is a multiple of 32 (llama.cpp's own fallback), else F32.
    Q2K,
}

impl Quant {
    /// The name `--quant` takes (`int8` and `int4` are accepted for Q8_0 and Q4_0 as well).
    pub fn parse(s: &str) -> Option<Quant> {
        Some(match s {
            "f32" => Quant::F32,
            "f16" | "fp16" => Quant::F16,
            "q8_0" | "int8" => Quant::Q8_0,
            "q4_0" | "int4" => Quant::Q4_0,
            "fp4" | "mxfp4" => Quant::Fp4,
            "q2" | "q2_k" => Quant::Q2K,
            _ => return None,
        })
    }

    /// The name recorded in a file's `ferrum.quant` metadata.
    pub fn tag(self) -> &'static str {
        match self {
            Quant::F32 => "f32",
            Quant::F16 => "f16",
            Quant::Q8_0 => "int8",
            Quant::Q4_0 => "int4",
            Quant::Fp4 => "fp4",
            Quant::Q2K => "q2",
        }
    }

    /// The GGUF tensor type of a matrix whose rows have `ne0` values.
    pub fn tensor_type(self, ne0: usize) -> u32 {
        let blocks = |k: usize| ne0.is_multiple_of(k);
        match self {
            Quant::F32 => GGML_F32,
            Quant::F16 => GGML_F16,
            Quant::Q8_0 if blocks(QK) => GGML_Q8_0,
            Quant::Q4_0 if blocks(QK) => GGML_Q4_0,
            Quant::Fp4 if blocks(QK) => GGML_MXFP4,
            Quant::Q2K if blocks(QK_K) => GGML_Q2_K,
            Quant::Q2K if blocks(QK) => GGML_Q4_0,
            _ => GGML_F32,
        }
    }
}

#[cfg(test)]
mod roundtrip_tests;
#[cfg(test)]
mod tests;
