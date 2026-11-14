//! Interoperability: models written by ferrum-gpu run in llama.cpp, Ollama and LM Studio (whose
//! engine is llama.cpp), and files llama.cpp writes run here.
//!
//! The real tools are used when they are installed and skipped, with the reason printed, when they
//! are not; the suite passes on a machine with none of them. Nothing here downloads a model: the
//! models are a few thousand parameters trained for a second on a sentence repeated 60 times.
//!
//! - `gguf`: the file's structure, checked against what llama.cpp's loader requires (no tool needed).
//! - `llamacpp`: tokenizer ids and greedy text agree with `llama-tokenize` / `llama-completion`
//!   for f32, Q8_0 and Q4_0 files; files `llama-quantize` writes are read here.
//! - `ollama`: `export-ollama` registers a model with a real `ollama serve` (private HOME and model
//!   directory, CPU only) and its raw greedy text equals ours.
//! - `lmstudio`: `export-lmstudio` runs the right `lms` commands (a fake `lms`: the real one writes
//!   outside the repository) on a file llama.cpp loads.

#![cfg(unix)]

mod gguf;
mod llamacpp;
mod lmstudio;
mod ollama;
mod support;
