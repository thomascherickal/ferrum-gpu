//! `ferrum-gpu lab`: one report about a machine's GPU. It checks that the GPU agrees with the CPU
//! reference (f32, Q8_0, Q4_0, and weights stored in blocks), measures decode and prefill speed per
//! precision, and prints the adapter, driver, features and memory budget. `--json` writes the same
//! as a file an owner of hardware we do not have can send back. The Studio's GPU lab tab, the
//! kernel benchmarks and the verification kit are all this command.

use crate::commands::flags::{switch, val, Args, Flag};
use crate::commands::{preflight, sysmem};
use ferrum_wgpu::Gpu;
use report::{AdapterFacts, Report};
use std::io::Write;
mod flags;
mod parity;
pub mod report;
mod speed;
pub use flags::*;
mod finish;
pub use finish::*;

#[cfg(test)]
mod tests;
