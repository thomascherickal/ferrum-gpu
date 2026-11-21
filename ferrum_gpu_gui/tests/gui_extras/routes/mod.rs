//! Routes and refusals the journey tests do not reach.
use crate::support::{app, app_with_model, call, call_raw, corpus, field, wait};
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;

mod a_model_cannot_be_replaced_while_a_run_is_going;
mod unknown_routes_and_methods_are_404_and_405_never_a_crash;
