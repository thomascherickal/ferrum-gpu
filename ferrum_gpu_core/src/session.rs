//! Choosing a device and holding the backend that was opened: what every front end needs before it
//! can train or generate.
use crate::engine::{Backend, CpuBackend, GpuBackend};
use ferrum_wgpu::device::select::AdapterSummary;
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::Gpu;
use std::sync::Arc;
mod devicechoice;
pub use devicechoice::*;
mod part;

#[cfg(test)]
mod tests;
