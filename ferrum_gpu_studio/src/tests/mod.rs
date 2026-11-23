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

mod flags_become_a_launch_or_an_error;
mod the_main_window_shows_the_page_of_the_in_process_server;
