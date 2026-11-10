#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! wgpu / WGSL engine for small Transformers. Hardware-free logic (selection, budget) is
//! pure and tested without a GPU.

pub mod bench_util;
pub mod budget;
pub mod device;
pub mod matmul;
pub mod select;
pub mod tensor;

pub use device::{Gpu, GpuError};
