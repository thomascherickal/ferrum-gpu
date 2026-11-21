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
