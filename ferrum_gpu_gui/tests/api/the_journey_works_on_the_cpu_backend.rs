use super::*;

#[test]
fn the_journey_works_on_the_cpu_backend() {
    let (srv, _app) = serve(cpu_app());
    journey(srv.addr.port());
    srv.stop();
}

#[test]
fn the_same_journey_works_on_the_gpu_backend() {
    let Some(g) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let app = App::with_backend(
        Arc::new(GpuBackend::new(Arc::new(g))),
        DeviceChoice::Gpu,
        None,
        ferrum_wgpu::Gpu::list(),
    );
    let (srv, _app) = serve(app);
    journey(srv.addr.port());
    let info = get(srv.addr.port(), "/api/info").json();
    assert_eq!(
        field(&field(&info, "device"), "active").as_str(),
        Some("gpu")
    );
    srv.stop();
}

#[test]
fn static_files_are_served_from_the_binary_and_nothing_else() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let page = get(p, "/");
    assert_eq!(page.status, 200);
    assert!(page.head.contains("text/html") && page.text().contains("<title>ferrum-gpu</title>"));
    assert!(get(p, "/index.html").text().contains("role=\"tablist\""));
    assert!(get(p, "/app.js").head.contains("text/javascript"));
    assert!(get(p, "/style.css").head.contains("text/css"));
    assert!(get(p, "/shaders/prelude.wgsl").text().contains("fn gid"));
    assert!(get(p, "/shaders/llama/rmsnorm.wgsl")
        .text()
        .contains("fn main"));
    for missing in [
        "/nope",
        "/shaders/llama/nope.wgsl",
        "/shaders/../Cargo.toml",
        "/shaders/llama/../fwd/add.wgsl",
        "/web/nothing.js",
        "/shaders/",
        "/etc/passwd",
        "/%2e%2e/Cargo.toml",
    ] {
        assert_eq!(get(p, missing).status, 404, "{missing}");
    }
    assert_eq!(call(p, "DELETE", "/api/info", b"").status, 405);
    assert_eq!(call(p, "POST", "/api/nothing", b"{}").status, 405);
    assert_eq!(call(p, "POST", "/", b"").status, 405);
    srv.stop();
}

#[test]
fn the_loopback_guard_and_request_limits_apply_through_the_real_server() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(
        raw(p, b"GET /api/info HTTP/1.1\r\nHost: evil.example\r\n\r\n").status,
        403
    );
    let cross = format!(
        "GET /api/info HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nOrigin: http://evil.example\r\n\r\n"
    );
    assert_eq!(raw(p, cross.as_bytes()).status, 403);
    assert_eq!(raw(p, b"junk\r\n\r\n").status, 400);
    let huge = format!("PUT /api/model.gguf HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nContent-Length: 999999999999\r\n\r\n");
    assert_eq!(raw(p, huge.as_bytes()).status, 413);
    srv.stop();
}
