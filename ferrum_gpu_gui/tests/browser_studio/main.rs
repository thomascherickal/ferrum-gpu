//! Drives the Studio tabs in headless Chrome against the real server: keyboard tab movement, Beginner/Expert, tool-tips, Datasets, the file picker, and real
//! Prepare and Tokenize jobs (the real commands, in-process). Skipped (with the reason printed) when
//! Chrome is missing, so the suite passes on a bare machine.

// the harness uses process groups, `kill`, `sh` and `timeout`: not built on Windows
#![cfg(unix)]

use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::start;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
#[path = "../common/mod.rs"]
mod common;

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

fn work(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_studio_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 30 short distinct documents in `dir/in`, and `dir/corpus.txt` with all of them.
fn corpus(dir: &Path) -> String {
    let inp = dir.join("in");
    std::fs::create_dir_all(&inp).unwrap();
    let mut all = String::new();
    for i in 0..30 {
        let doc =
            format!("document {i}: the quick brown fox jumps over the lazy dog, again and again. ");
        let doc = doc.repeat(8);
        std::fs::write(inp.join(format!("d{i:02}.txt")), &doc).unwrap();
        all.push_str(&doc);
    }
    std::fs::write(dir.join("corpus.txt"), &all).unwrap();
    inp.display().to_string()
}

mod the_new_tabs_work_in_a_real_browser_with_the_real_commands;
