use super::*;

#[test]
fn preview_returns_text_and_validates_bytes() {
    let d = tmp();
    let f = put(&d, "t.txt", b"hello\nworld\n");
    let p = f.to_str().unwrap();
    let (s, j) = get("/api/fs/preview", &[("path", p)]);
    assert_eq!(s, 200);
    assert_eq!(field(&j, "text").as_str(), Some("hello\nworld\n"));
    assert_eq!(field(&j, "lines").as_f64(), Some(2.0));
    assert_eq!(field(&j, "binary"), Json::Bool(false));
    let (_, j) = get("/api/fs/preview", &[("path", p), ("bytes", "5")]);
    assert_eq!(field(&j, "text").as_str(), Some("hello"));
    assert_eq!(field(&j, "truncated"), Json::Bool(true));
    for bad in ["0", "-1", "x", ""] {
        assert_eq!(
            get("/api/fs/preview", &[("path", p), ("bytes", bad)]).0,
            400,
            "{bad:?}"
        );
    }
    let (s, j) = get("/api/fs/preview", &[("path", d.to_str().unwrap())]);
    assert_eq!(s, 400);
    assert!(field(&j, "error").as_str().unwrap().contains("not a file"));
}

#[test]
fn other_methods_and_unknown_routes_are_refused() {
    assert_eq!(call("POST", "/api/fs/list", &[]).0, 405);
    assert_eq!(call("DELETE", "/api/fs/stat", &[]).0, 405);
    assert_eq!(get("/api/fs/nope", &[]).0, 404);
}

#[test]
fn the_api_router_sends_fs_paths_here() {
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
        path: "/api/fs/list".into(),
        query: vec![],
        headers: vec![],
        body: vec![],
    };
    assert_eq!(crate::api::route(&a, &req).status, 200);
}
