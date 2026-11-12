#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! wgpu / WGSL engine for small Transformers. Hardware-free logic (selection, budget) is
//! pure and tested without a GPU.
//!
//! Layout: [`device`] (adapter, runtime, buffers, budget), [`ops`] (kernels: forward, backward,
//! optimizer, QAT, quantised, Llama), [`models`] (inference), [`training`] (trainers). The older
//! flat names (`ops_bwd`, `gpt2`, `train`, `tensor`, ...) are re-exported below.

pub mod bench_util;
pub mod device;
pub mod models;
pub mod ops;
pub mod shaders;
pub mod tolerance;
pub mod training;

pub use device::{budget, runtime, select, tensor};
pub use device::{Gpu, GpuError};
pub use models::{gpt2, llama};
pub use ops::{
    bwd as ops_bwd, dropout as ops_dropout, llama as ops_llama, llama_bwd as ops_llama_bwd, matmul,
    opt as ops_opt, qat as ops_qat, quant as ops_quant,
};
pub use training::{gpt2 as train, llama as llama_train};
