//! What happens around each training step: progress lines, the metrics CSV, the stop conditions
//! (`--max-seconds`, a `<model>.stop` file, early stopping), checkpoints and validation.
use super::score::mean_loss;
use ferrum_model::checkpoint::{atomic_write, to_bytes};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Schedule;
use ferrum_ref::trainer::{Checkpoint, Hooks, StepStats};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
mod validation;
pub use validation::*;
mod part;

#[cfg(test)]
mod tests;
