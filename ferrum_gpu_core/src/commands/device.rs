//! The `--device auto` rule, from the measured crossovers (`docs/train-speed.md`: batched
//! training is 1.3x to 5.5x faster on the GPU; `docs/decode-crossover.md`: single-token decode is
//! not faster on the GPU up to 34M parameters). A pure function over an adapter table, so tests
//! use fake adapters.
use crate::session::DeviceChoice;
use ferrum_wgpu::device::select::{choose, AdapterSummary};
mod workload;
pub use workload::*;

#[cfg(test)]
mod tests;
