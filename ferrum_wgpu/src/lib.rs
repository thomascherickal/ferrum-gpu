#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! wgpu / WGSL engine for small Transformers. Hardware-free logic (selection, budget) is
//! pure and tested without a GPU.

pub mod bench_util;
pub mod budget;
pub mod device;
pub mod gpt2;
pub mod llama;
pub mod llama_train;
pub mod matmul;
pub mod ops;
pub mod ops_bwd;
pub mod ops_llama;
pub mod ops_llama_bwd;
pub mod ops_opt;
pub mod ops_qat;
pub mod ops_quant;
pub mod runtime;
pub mod select;
pub mod tensor;
pub mod tolerance;
pub mod train;

pub use device::{Gpu, GpuError};
