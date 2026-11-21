//! The backend-independent training loop: run configuration, checkpoints, hooks, `train`.
use super::engine::{Engine, Snapshot, StepConfig, StepStats};
use crate::optim::{Adam, Schedule};
use crate::rng::Rng;
mod tokensource;
pub use tokensource::*;
mod runshape;
pub use runshape::*;
mod report;
pub use report::*;
mod batch;
use batch::{load_batch, step_config};
mod train;
pub use train::*;
