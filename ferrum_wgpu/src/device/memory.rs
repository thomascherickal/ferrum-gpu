//! How much device memory a model may use: the budget the pre-flight checks a plan against.
//!
//! `wgpu` has no portable memory query, so the numbers come from outside (the injected readers in
//! `ferrum_gpu_core`), and this module only decides, purely, what they mean. A budget is always an
//! **estimate** and says where it came from ([`Source`]).
//!
//! The headroom is taken from measurement, not guessed. On the NVIDIA MX330 (2048 MiB total, 1989 MiB
//! free) memory plans of up to 1449 MiB (73% of the free memory) loaded and ran, and plans of 1721 MiB
//! (86%) and above failed with "GPU out of memory" (`capability.md`, 2026-11-14): the plan counts the
//! resident buffers, while loading and the driver need room beyond them. So a discrete card's usable
//! budget is [`VRAM_USABLE_PERCENT`] of its **free** memory. Integrated and unified-memory adapters
//! share the machine's RAM with everything else, so their budget is a share of the RAM that is still
//! available ([`UNIFIED_USABLE_PERCENT`], an assumption: it was not measured on unified-memory hardware).
use super::budget::{Budget, Source};
mod vram_usable_percent;
pub use vram_usable_percent::*;
mod describe;
pub use describe::*;

#[cfg(test)]
mod tests;
