//! Helpers: an app on the CPU backend, in-process API calls, and a training run to completion.
use ferrum_gpu_gui::api::route;
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::http::Request;
use ferrum_gpu_gui::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};
mod app;
pub(crate) use app::*;
mod trained;
pub(crate) use trained::*;
