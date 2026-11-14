//! Runs against real tools that use the GPU or change things outside the repository. All opt-in
//! (they would surprise a developer running `cargo test`), and each logs every step through
//! [`crate::support::log::step`]; set `FERRUM_INTEROP_LOG=file` to keep the record.
//!
//! - `FERRUM_INTEROP_GPU=1`: llama.cpp's Vulkan build and Ollama (CUDA and Vulkan) on the GPUs.
//! - `FERRUM_INTEROP_LMS=1`: a real `lms import` into the user's LM Studio library (under
//!   `~/.lmstudio`), removed again afterwards.

mod llamacpp_gpu;
mod lms;
mod ollama_gpu;
