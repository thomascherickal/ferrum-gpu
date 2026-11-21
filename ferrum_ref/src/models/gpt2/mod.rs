//! GPT-2 weights (plain host f32, matrices stored `[in, out]` row-major) and the CPU oracle
//! forward pass the GPU decoder is checked against.

pub use decode::Cache;
pub use params::{is_matrix_in_block, param_lens};

use crate::matmul::{matmul, rand_vec};
use crate::nn;
mod config;
mod decode;
mod params;
pub use config::*;
mod part;

#[cfg(test)]
mod tests;
