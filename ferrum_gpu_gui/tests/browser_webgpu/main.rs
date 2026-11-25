//! Phase 5b: the in-browser WebGPU runtime against the CPU oracle, and the browser-only JS parts.
//! Both skip, with the reason printed, when Node or Chrome (or WebGPU in Chrome) is missing.

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
#[path = "../common/mod.rs"]
mod common;

fn app() -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ))
}

mod the_js_reader_repack_and_rope_tables_match_the_rust_side;
