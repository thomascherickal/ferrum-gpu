// Provenance: semantics (RMSNorm, RoPE NORM/NEOX pairing and frequency scaling, GQA attention,
// SwiGLU) copied from ferrum-cpu `ferrum_core/src/llm.rs` (commit 56a9d3a). Rewritten over plain
// slices with host-built cos/sin tables so the GPU kernels can share the exact angles; ours from
// here on.
//! Forward ops of the Llama / Qwen stack on row-major f32 slices: the parity oracle for the
//! GPU kernels.

mod ropetype;
pub use ropetype::*;
mod rope_tables;
pub use rope_tables::*;

#[cfg(test)]
mod tests;
