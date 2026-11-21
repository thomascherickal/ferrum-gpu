//! GPT-2 weights (plain host f32, matrices stored `[in, out]` row-major) and the CPU oracle
//! forward pass the GPU decoder is checked against.

pub use decode::Cache;
pub use params::{is_matrix_in_block, param_lens};

use crate::matmul::{matmul, rand_vec};
use crate::nn;
mod config;
pub use config::*;
mod weights;
pub use weights::*;
mod decode;
mod forward;
mod head;
mod params;

#[cfg(test)]
mod tests;
