//! Parity checks: the GPU against the CPU reference on a small random model. Needs a device, so
//! the arithmetic (what counts as a pass) lives in `report`; this file only runs the models.
use super::report::Check;
use super::text;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_wgpu::llama::{dequantised, GpuLlama};
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::tolerance::{max_rel_err, FORWARD};
use ferrum_wgpu::Gpu;
mod config;
use config::*;
mod run;
pub use run::*;

#[cfg(test)]
mod tests;
