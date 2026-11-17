//! Tests of the app's wiring on Tauri's mock runtime (no display needed): the window is created on the
//! in-process server's page, that page is served with its policy, and stopping is clean.

use super::*;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use tauri::test::{mock_builder, mock_context, noop_assets};

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn cpu_server() -> Running {
    start_server(&args("--device cpu --threads 1")).expect("the server starts")
}

/// A raw GET of `/` with the right Host header; the whole response as text.
fn get(port: u16, path: &str) -> String {
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(
        c,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn the_main_window_shows_the_page_of_the_in_process_server() {
    let server = cpu_server();
    let url = server.url();
    let mut app = configure(mock_builder(), server)
        .build(mock_context(noop_assets()))
        .expect("the app builds");
    // Tauri runs the setup hook (which creates the window) on the first event-loop iteration
    #[allow(deprecated)]
    app.run_iteration(|_, _| {});
    let w = app
        .get_webview_window("main")
        .expect("the main window exists");
    assert_eq!(w.url().unwrap().as_str(), url);
    // (the mock runtime does not keep window titles, so TITLE is only checked to be a real name)
    assert!(TITLE.contains("Studio") && !TITLE.is_empty());
    let studio = app.state::<Studio>();
    assert_eq!(studio.url, url);
    assert!(studio.running());
    studio.stop();
}

#[test]
fn that_page_is_the_studio_with_its_policy() {
    let server = cpu_server();
    let port = server.addr.port();
    let page = get(port, "/");
    assert!(
        page.starts_with("HTTP/1.1 200"),
        "{}",
        &page[..40.min(page.len())]
    );
    assert!(
        page.contains("Content-Security-Policy"),
        "the policy is on the page"
    );
    assert!(
        page.contains("Datasets") && page.contains("ui-mode"),
        "the Studio page"
    );
    server.stop();
}

#[test]
fn the_server_refuses_a_foreign_host_so_the_window_is_the_only_client() {
    let server = cpu_server();
    let port = server.addr.port();
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(
        c,
        "GET / HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 403"), "{out}");
    server.stop();
}

#[test]
fn stopping_ends_the_server_and_a_second_stop_does_nothing() {
    let server = cpu_server();
    let port = server.addr.port();
    let studio = Studio::new(server);
    assert!(studio.running());
    studio.stop();
    assert!(!studio.running());
    studio.stop();
    assert!(
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().unwrap(),
            Duration::from_millis(500)
        )
        .is_err(),
        "nothing listens any more"
    );
}

#[test]
fn flags_become_a_launch_or_an_error() {
    assert!(matches!(launch(&[]), Ok(Launch::Start(a)) if a.is_empty()));
    assert!(matches!(launch(&args("--device cpu")), Ok(Launch::Start(a)) if a.len() == 2));
    assert_eq!(launch(&args("--help")), Ok(Launch::Help(USAGE)));
    assert_eq!(launch(&args("-h")), Ok(Launch::Help(USAGE)));
    for bad in [
        "--device tpu",
        "--port x",
        "--nope",
        "--threads 0",
        "--device",
    ] {
        assert!(launch(&args(bad)).is_err(), "{bad}");
    }
}

#[test]
fn a_bad_device_flag_stops_before_anything_starts() {
    assert!(start_server(&args("--device tpu")).is_err());
    assert!(start_server(&args("--port 70000")).is_err());
    // asking for a GPU adapter that does not exist is an error, not a silent CPU
    assert!(start_server(&args("--device gpu --gpu-adapter no-such-adapter-zzz")).is_err());
}

#[test]
fn auto_falls_back_to_the_cpu_when_the_named_adapter_is_absent() {
    let s = start_server(&args(
        "--device auto --gpu-adapter no-such-adapter-zzz --threads 1",
    ))
    .expect("auto never fails for want of a GPU");
    let page = get(s.addr.port(), "/api/info");
    assert!(page.contains("\"active\":\"cpu\""), "{page}");
    s.stop();
}
