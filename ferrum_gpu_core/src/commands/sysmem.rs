//! Reading the machine's memory for the pre-flight. The reads are behind [`Readers`] so tests use
//! fixtures and never run `nvidia-smi` or touch `/proc`; [`System`] is the real implementation and the
//! only part that needs real hardware. What the numbers mean is decided in `ferrum_wgpu::memory`.
use ferrum_wgpu::budget::{parse_nvidia_smi_free, sysfs_vram_mb, Budget};
use ferrum_wgpu::memory::{resolve_memory, MemoryFacts};
use ferrum_wgpu::select::{AdapterSummary, Kind};
use std::path::Path;
mod readers;
pub use readers::*;
mod facts_for;
pub use facts_for::*;

#[cfg(test)]
mod tests;
