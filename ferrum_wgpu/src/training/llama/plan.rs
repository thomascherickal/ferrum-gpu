//! Buffer plans of the Llama trainer: every device buffer, named and sized, so the footprint is
//! checked against the budget before anything is allocated.
use crate::budget::BufferPlan;
use crate::ops_opt::parts_for;
use ferrum_ref::llama::{Config, Llama};
mod f;
pub use f::*;
mod activation_plan;
pub use activation_plan::*;
