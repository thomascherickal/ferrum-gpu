//! Drives the real page in headless Chrome: create a model, train, generate. Skipped (with the
//! reason printed) when no Chrome is installed, so the suite still passes on a bare machine.

mod common;

use gpu_gui::app::{App, DeviceChoice};
use gpu_gui::engine::CpuBackend;
use gpu_gui::json::Json;
use gpu_gui::server::start;
use std::sync::Arc;
use std::time::Duration;

fn fetch_result(port: u16) -> Option<String> {
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
fn the_page_trains_and_generates_in_a_real_browser() {
    let Some(chrome) = common::chrome() else {
        eprintln!("SKIP: no Chrome or Chromium installed");
        return;
    };
    let app = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let server = start(app, 0).unwrap();
    let port = server.addr.port();
    let browser = common::Browser::open(
        &chrome,
        &format!("http://127.0.0.1:{port}/?autotest=ui"),
        "ui",
    );
    let Some(result) = common::wait_for(Duration::from_secs(60), || fetch_result(port)) else {
        panic!(
            "the page never reported a result; browser console:\n{}",
            browser.console()
        );
    };
    eprintln!("page reported: {result}");
    let j = Json::parse(&result).unwrap();
    assert_eq!(j.get("kind").and_then(Json::as_str), Some("ui"));
    assert_eq!(j.get("pass"), Some(&Json::Bool(true)), "{result}");
    assert!(j
        .get("device")
        .and_then(Json::as_str)
        .unwrap()
        .starts_with("CPU"));
    server.stop();
}
