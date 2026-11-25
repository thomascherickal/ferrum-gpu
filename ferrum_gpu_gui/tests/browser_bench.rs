//! In-browser decode speed on every WebGPU adapter Chrome offers. Slow (minutes), so it only runs
//! on request: `cargo test -p ferrum_gpu_gui --test browser_bench -- --ignored --nocapture`.

mod common;

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::sync::Arc;
use std::time::Duration;

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
    let Some(text) = common::wait_for(Duration::from_secs(1500), || common::selftest_result(port))
    else {
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
