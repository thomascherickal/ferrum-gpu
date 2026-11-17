#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! Local web GUI for ferrum-gpu (Phase 5a) and the host page for in-browser WebGPU (Phase 5b).
//! Everything is std-only: a small HTTP server, a JSON reader/writer and a byte tokenizer.

pub mod api;
pub mod app;
pub mod assets;
pub mod fsapi;
pub mod jobs;
pub mod selftest;
pub mod server;

/// The shared core, re-exported under the paths this crate has always had.
pub use ferrum_gpu_core::{cancel, engine, http, json, stop, tokenizer};
