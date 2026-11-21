//! The GPU backend: `ferrum_wgpu`'s resident Llama decoder and trainer.
use super::{
    generate_with, run_training, Backend, Engine, EngineOptions, GenOpts, Gpt2Feed, Progress,
    TrainSpec,
};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::llama::Llama;
use ferrum_wgpu::gpt2::GpuGpt2;
use ferrum_wgpu::llama::GpuLlama;
use ferrum_wgpu::llama_train::GpuLlamaTrainer;
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::train::GpuTrainer;
use ferrum_wgpu::Gpu;
use std::sync::{Arc, Mutex};
mod gpubackend;
pub use gpubackend::*;
mod backend;
