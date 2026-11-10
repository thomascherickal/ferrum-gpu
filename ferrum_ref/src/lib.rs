#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! CPU reference kernels: the parity oracle for the GPU engine. Grows phase by phase.

pub mod gpt2;
pub mod grad;
pub mod matmul;
pub mod nn;
pub mod tuned;
