use super::*;

#[test]
fn the_api_router_sends_gguf_paths_here() {
    use crate::app::{App, DeviceChoice};
    use crate::engine::CpuBackend;
    let a = std::sync::Arc::new(App::with_backend(
        std::sync::Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let req = Request {
        method: "GET".into(),
        path: "/api/gguf/models".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
}
