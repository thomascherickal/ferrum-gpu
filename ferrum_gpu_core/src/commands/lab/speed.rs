//! Speed per precision on a random-weight model: decode (one token at a time with the KV cache) and
//! prefill (a whole sequence at once). Random weights give the same speed as trained ones, but not
//! text, and the model is small, so this measures launch and dispatch cost more than bandwidth;
//! the report says so.
use super::report::Speed;
use super::text;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_wgpu::bench_util::time_ms;
use ferrum_wgpu::llama::{plan, GpuLlama};
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::Gpu;
mod size;
pub(super) use size::*;
mod measure;
pub(super) use measure::*;

#[cfg(test)]
mod tests;
