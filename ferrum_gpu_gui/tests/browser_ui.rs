//! Drives the real page in headless Chrome: create a model, train, generate. Skipped (with the
//! reason printed) when no Chrome is installed, so the suite still passes on a bare machine.

mod common;

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::sync::Arc;
use std::time::Duration;

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
        180,
    );
    let Some(result) = common::wait_for(Duration::from_secs(60), || common::selftest_result(port))
    else {
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
