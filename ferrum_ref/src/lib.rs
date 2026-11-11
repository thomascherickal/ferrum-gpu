#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! CPU reference kernels: the parity oracle for the GPU engine.
//!
//! Layout: [`kernels`] (matmul, forward ops, RNG), [`models`] (GPT-2, Llama/Qwen2), [`training`]
//! (gradients, AdamW, QAT, the loop). The flat names (`nn`, `grad`, `gpt2`, ...) are re-exported.

pub mod kernels;
pub mod models;
pub mod training;

pub use kernels::{matmul, nn, rng, tuned};
pub use models::{gpt2, llama};
pub use training::{grad, optim, qat, trainer};
