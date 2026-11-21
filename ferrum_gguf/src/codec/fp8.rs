//! 8-bit floats for GPU weight storage: E4M3 (the OCP "fn" variant: bias 7, no infinities, the
//! all-ones code is NaN, largest value 448) and E5M2 (IEEE-like: bias 15, infinities and NaN,
//! largest finite value 57344). llama.cpp has no FP8 tensor type, so these never go into a GGUF
//! file; `ferrum-gpu` converts a model to them when it loads it onto the device.
//!
//! A block is 32 values with an f16 scale (the largest magnitude divided by the format's largest
//! value) and one code byte each: 34 bytes, laid out like Q8_0 so one packer serves both.
use crate::f16::{f16_to_f32, f32_to_f16};
use crate::QK;
mod block;
pub use block::*;
mod encode;
mod part;
pub use encode::*;

#[cfg(test)]
mod tests;
