use super::*;
use crate::app::DeviceChoice;
use crate::engine::CpuBackend;
use crate::test_support::{field, reply, request};
use ferrum_wgpu::select::{AdapterFeatures, AdapterSummary, Kind};
use std::sync::Arc;

fn app(adapters: Vec<AdapterSummary>) -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        adapters,
    ))
}

fn get(a: &App, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    reply(route(a, &request("GET", path, query, &[])))
}

mod bad_inputs_are_400s_that_name_the_field;
mod only_these_paths_are_handled;
mod other_methods_are_refused_and_the_router_sends_these_paths_here;
