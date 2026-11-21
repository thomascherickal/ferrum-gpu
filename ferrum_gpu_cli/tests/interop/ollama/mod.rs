//! Ollama on the CPU: `export-ollama` registers a model with a real `ollama serve`, and Ollama's raw
//! greedy text equals ours. The server gets a private HOME and model directory (it writes an
//! identity key and its blobs there, never into the user's own) and is kept off the GPU. The GPU
//! runs are in `real/ollama_gpu.rs`.
use crate::skip;
use crate::support::model::{ferrum, fixture, ours, PROMPT};
use crate::support::ollama_server::{free_port, http, start, Mode};
use crate::support::tools::ollama;
use ferrum_gpu_core::json::Json;
use std::process::Command;

mod export_ollama_registers_the_model_and_ollama_continues_it_like_we_do;
mod export_ollama_says_the_server_is_not_running_when_it_is_not;
