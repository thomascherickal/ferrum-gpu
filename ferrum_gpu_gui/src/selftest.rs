//! Parity vectors for the in-browser runtime (Phase 5b): small seeded Llama / Qwen2 models as GGUF
//! bytes, a token sequence and the CPU oracle's logits for it. The browser loads the same bytes,
//! runs its WebGPU kernels and compares. For quantised files the oracle is the model decoded
//! from those very bytes, so only the kernels (not the quantiser) are being tested.
use crate::json::Json;
use ferrum_gguf::Gguf;
use ferrum_model::gguf::gpt2::Quant;
use ferrum_model::gguf::llama::{from_gguf, to_gguf};
use ferrum_ref::llama::{rope_tables, Config, Llama, RopeScaling, RopeType};
use ferrum_wgpu::ops::quant::{pack, QKind};
mod b64;
pub use b64::*;
mod probe_m;
pub use probe_m::*;

#[cfg(test)]
mod tests;
