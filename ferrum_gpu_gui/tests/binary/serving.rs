use crate::support::{get, serve, spawn};
use std::io::BufRead;

#[test]
fn it_serves_the_page_and_the_api_on_a_free_port() {
    let s = serve();
    let page = get(s.port, "/");
    assert!(page.starts_with("HTTP/1.1 200") && page.contains("<title>ferrum-gpu</title>"));
    let info = get(s.port, "/api/info");
    assert!(info.contains("\"active\":\"cpu\""), "{info}");
}

#[test]
fn it_keeps_serving_after_its_stdout_is_closed() {
    // `serve` dropped the reader, so the pipe is closed; the binary prints "press Ctrl-C" at
    // about that time and must not die of a broken pipe
    let s = serve();
    std::thread::sleep(std::time::Duration::from_millis(400));
    assert!(get(s.port, "/api/info").starts_with("HTTP/1.1 200"));
}

#[test]
fn a_port_that_is_taken_exits_with_1_and_says_so() {
    let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port().to_string();
    let o = spawn(&["--port", &port, "--device", "cpu"])
        .wait_with_output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.is_empty(), "the reason is on stderr");
}

#[test]
fn the_first_lines_name_the_device_and_the_address() {
    let mut child = spawn(&["--port", "0", "--device", "cpu", "--threads", "1"]);
    let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let first = lines.next().unwrap().unwrap();
    assert!(
        first.starts_with("device: CPU") && first.ends_with("(cpu)"),
        "{first}"
    );
    let second = lines.next().unwrap().unwrap();
    assert!(
        second.starts_with("ferrum-gpu GUI: http://127.0.0.1:"),
        "{second}"
    );
    child.kill().ok();
    child.wait().ok();
}

#[test]
fn with_a_serve_time_it_serves_then_exits_normally() {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_ferrum_gpu_gui"))
        .args(["--port", "0", "--device", "auto", "--threads", "1"])
        .env("FERRUM_SERVE_SECONDS", "1")
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert_eq!(o.status.code(), Some(0), "{out}");
    assert!(
        out.contains("ferrum-gpu GUI: http://127.0.0.1:") && out.contains("press Ctrl-C"),
        "{out}"
    );
}

#[test]
fn an_auto_device_that_falls_back_to_the_cpu_says_why_before_the_address() {
    let mut child = spawn(&[
        "--port",
        "0",
        "--device",
        "auto",
        "--gpu-adapter",
        "no-such-adapter-xyz",
    ]);
    let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let device = lines.next().unwrap().unwrap();
    let note = lines.next().unwrap().unwrap();
    child.kill().ok();
    child.wait().ok();
    assert!(device.starts_with("device: CPU"), "{device}");
    assert!(note.starts_with("note: no usable GPU"), "{note}");
}
