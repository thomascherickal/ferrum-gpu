//! Memory pre-flight (6.4): list every device buffer a run will create, using the same plans the
//! GPU engine checks itself, and refuse before anything is opened or uploaded.
use ferrum_ref::llama::Llama;
use ferrum_wgpu::budget::{Budget, BufferPlan};
use ferrum_wgpu::llama::plan as decode_plan;
use ferrum_wgpu::llama_train::{activation_plan, static_plan};
use ferrum_wgpu::select::AdapterSummary;
mod train_plan;
pub use train_plan::*;

#[cfg(test)]
mod tests;
