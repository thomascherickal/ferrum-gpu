//! Parameter shape arithmetic, deterministic random weights and the tied-head helper.
use super::{Block, Config, Llama, RopeScaling};
mod lens;
pub use lens::*;
mod block_params;
mod random_init;
pub use block_params::*;
mod param_list;
