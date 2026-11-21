//! `/api/info` tells the page what each adapter can do (its features) and how much memory a model may
//! use (a budget with its source), using a fake adapter list so no real hardware is involved.
use crate::support::{call, field};
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_wgpu::select::{AdapterFeatures, AdapterSummary, Kind};
use std::sync::Arc;

fn adapter(name: &str, kind: Kind, features: AdapterFeatures) -> AdapterSummary {
    AdapterSummary {
        name: name.into(),
        backend: "Vulkan".into(),
        kind,
        max_buffer: 1 << 32,
        max_binding: 2047 << 20,
        max_workgroup_invocations: 256,
        features,
    }
}

fn app_with(adapters: Vec<AdapterSummary>) -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        adapters,
    ))
}

fn adapters_of(j: &Json) -> Vec<Json> {
    match field(j, "adapters") {
        Json::Arr(a) => a,
        other => panic!("adapters is not a list: {other}"),
    }
}

mod each_adapter_lists_its_features_by_name;
