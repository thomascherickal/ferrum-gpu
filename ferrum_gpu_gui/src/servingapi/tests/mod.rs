use super::*;
use crate::app::DeviceChoice;
use crate::engine::CpuBackend;
use std::sync::Arc;

fn app() -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ))
}

fn call(a: &App, method: &str, path: &str, body: &[u8]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: vec![],
        headers: vec![],
        body: body.to_vec(),
    };
    let r = route(a, &req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

mod a_server_starts_through_the_api_serves_and_stops;
mod only_the_serve_and_tools_paths_are_handled;
mod the_api_router_sends_these_paths_here;
mod tools_are_listed_with_a_path_or_the_reason_they_are_missing;
