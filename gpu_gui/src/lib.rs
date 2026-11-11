#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! Local web GUI for ferrum-gpu (Phase 5a) and the host page for in-browser WebGPU (Phase 5b).
//! Everything is std-only: a small HTTP server, a JSON reader/writer and a byte tokenizer.

pub mod api;
pub mod app;
pub mod assets;
pub mod engine;
pub mod http;
pub mod json;
pub mod server;
pub mod tokenizer;
