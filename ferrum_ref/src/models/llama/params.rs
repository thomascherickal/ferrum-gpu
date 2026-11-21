//! Parameter shape arithmetic, deterministic random weights and the tied-head helper.
use super::{Block, Config, Llama, RopeScaling};
use crate::matmul::rand_vec;
mod param_lens_m;
pub use param_lens_m::*;
mod part;
