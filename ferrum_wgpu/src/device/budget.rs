// Provenance: the nvidia-smi / sysfs parsing idea is from ferrum-cpu
// `ferrum_gui/src/sizing/hardware.rs` (commit 56a9d3a); rewritten here, ours from now on.
//! GPU memory budget (wgpu has no portable VRAM query) and the plan-then-allocate check.
use std::path::Path;
mod source;
pub use source::*;
mod bufferplan;
pub use bufferplan::*;

#[cfg(test)]
mod tests;
