//! In-browser decode speed on every WebGPU adapter Chrome offers. Slow (minutes), so it only runs
//! on request: `cargo test -p ferrum_gpu_gui --test browser_bench -- --ignored --nocapture`.

mod common;

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::sync::Arc;
use std::time::Duration;

fn result(port: u16) -> Option<String> {
    use std::io::{Read, Write};
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        c,
        "GET /api/selftest/result HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n"
    )
    .ok()?;
    let mut out = String::new();
    c.read_to_string(&mut out).ok()?;
    let (head, body) = out.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_string())
}

#[test]
#[ignore = "slow: measures decode speed in a real browser"]
fn measure_in_browser_decode_speed() {
    let Some(chrome) = common::chrome() else {
        eprintln!("SKIP: no Chrome or Chromium installed");
        return;
    };
    let app = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let server = start(app, 0).unwrap();
    let port = server.addr.port();
    let browser = common::Browser::open(
        &chrome,
        &format!("http://127.0.0.1:{port}/?autotest=bench"),
        "bench",
        1800,
    );
    let Some(text) = common::wait_for(Duration::from_secs(1500), || result(port)) else {
        panic!("no result; console:\n{}", browser.console());
    };
    let j = Json::parse(&text).unwrap();
    if j.get("skipped") == Some(&Json::Bool(true)) {
        eprintln!(
            "SKIP: {}",
            j.get("reason").and_then(Json::as_str).unwrap_or("?")
        );
        return;
    }
    println!("BENCH {text}");
    assert_eq!(j.get("pass"), Some(&Json::Bool(true)), "{text}");
    server.stop();
}
