//! The GUI's API end to end over real TCP: the same flow on the CPU and the GPU backend.
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::{CpuBackend, GpuBackend};
use ferrum_gpu_gui::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod helpers;
mod http;
mod journey;
use helpers::*;
use http::*;
use journey::*;

mod a_file_that_is_not_a_model_is_refused_on_import;
mod a_run_can_be_stopped_and_keeps_its_partial_weights;
mod bad_input_gets_a_clear_error_and_never_a_crash;
mod hostile_paths_and_slow_clients_get_a_clear_status;
mod info_describes_the_device_the_models_and_the_limits;
mod many_clients_load_the_api_at_once;
mod models_move_through_gguf_in_every_precision;
mod the_journey_works_on_the_cpu_backend;
