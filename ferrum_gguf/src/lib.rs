#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! GGUF v3 container: parse, dequantise to f32, and build files. std only.
//!
//! Format knowledge (magic, value tags, tensor type ids, block layouts of Q8_0 and Q4_0) follows
//! the public GGUF/ggml specification and was cross-checked against ferrum-cpu's reader
//! (`ferrum_core/src/gguf/`, commit 56a9d3a); the code here is our own.

pub mod codec;
pub mod io;

pub use codec::{blocks, f16, quant};
pub use io::{reader, writer};

pub use reader::{Gguf, GgufError, TensorInfo, Value};
pub use writer::GgufBuilder;

/// GGUF magic, little-endian "GGUF".
pub const MAGIC: u32 = 0x4655_4747;
/// Default tensor-data alignment.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml tensor type: 32-bit float.
pub const GGML_F32: u32 = 0;
/// ggml tensor type: 16-bit float.
pub const GGML_F16: u32 = 1;
/// ggml tensor type: 16-bit signed integer (token files).
pub const GGML_I16: u32 = 25;
/// ggml tensor type: 32-bit signed integer (token files).
pub const GGML_I32: u32 = 26;
/// ggml tensor type: 4-bit blocks of 32 with an f16 scale.
pub const GGML_Q4_0: u32 = 2;
/// ggml tensor type: 4-bit blocks of 32 with an f16 scale and min (read only).
pub const GGML_Q4_1: u32 = 3;
/// ggml tensor type: 5-bit blocks of 32 with an f16 scale (read only).
pub const GGML_Q5_0: u32 = 6;
/// ggml tensor type: 5-bit blocks of 32 with an f16 scale and min (read only).
pub const GGML_Q5_1: u32 = 7;
/// ggml tensor type: 8-bit blocks of 32 with an f16 scale and sum (read only).
pub const GGML_Q8_1: u32 = 9;
/// ggml tensor type: 2-bit super-blocks of 256 (read only).
pub const GGML_Q2_K: u32 = 10;
/// ggml tensor type: 3-bit super-blocks of 256 (read only).
pub const GGML_Q3_K: u32 = 11;
/// ggml tensor type: 4-bit super-blocks of 256 (read only).
pub const GGML_Q4_K: u32 = 12;
/// ggml tensor type: 5-bit super-blocks of 256 (read only).
pub const GGML_Q5_K: u32 = 13;
/// ggml tensor type: 6-bit super-blocks of 256 (read only).
pub const GGML_Q6_K: u32 = 14;
/// ggml tensor type: 8-bit blocks of 32 with an f16 scale.
pub const GGML_Q8_0: u32 = 8;
/// Values per quantisation block.
pub const QK: usize = 32;
/// Values per k-quant super-block.
pub const QK_K: usize = 256;
