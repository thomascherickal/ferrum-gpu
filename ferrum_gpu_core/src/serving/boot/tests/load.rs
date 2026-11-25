//! Many clients at once against the real server around a tiny GPT-2 on the CPU backend: every
//! request gets a well-formed answer (a result, or the documented 503 past the in-flight cap), none
//! takes longer than a generous bound, and the server is healthy afterwards. The `LOAD` line goes
//! to system-profile.md through `scripts/load_profile.py`. The heavier run, on request:
//! `cargo test --release -p ferrum_gpu_core --lib -- --ignored heavy_load --nocapture`
use super::*;
use crate::json::Json;
use ferrum_test_support::load::{ask, drive, summary};
use std::net::SocketAddr;
use std::time::Duration;

const BODY: &str = r#"{"requests":[{"prompt":"hi","max_tokens":2}]}"#;

fn generate(addr: SocketAddr) -> Vec<u8> {
    format!(
        "POST /v1/generate HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\n\r\n{BODY}",
        BODY.len()
    )
    .into_bytes()
}

fn run(name: &str, clients: usize, each: usize) {
    let d = tmp(name);
    let model = gpt2_file(&d);
    let b = boot(
        &argv(&format!("--model {model} --port 0 --device cpu")),
        &mut Vec::new(),
    )
    .unwrap();
    let addr = b.addr;
    let (answers, wall) = drive(addr, clients, each, |_| generate(addr));
    assert_eq!(answers.len(), clients * each);
    for a in &answers {
        match a.status {
            200 => assert!(Json::parse(&a.body).is_ok(), "{}", a.body),
            503 => assert_eq!(a.body, r#"{"error":"too many requests in flight"}"#),
            other => panic!("status {other}: {}", a.body),
        }
        assert!(a.took < Duration::from_secs(60), "{:?}", a.took);
    }
    println!("{}", summary(name, clients, &answers, wall));
    let health = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    assert_eq!(ask(addr, health.as_bytes()).status, 200);
    assert_eq!(ask(addr, &generate(addr)).status, 200);
    b.server.stop();
}

#[test]
fn many_clients_get_an_answer_or_a_clear_refusal() {
    run("json-server", 48, 4);
}

#[test]
#[ignore = "heavy: 3,840 requests from 96 clients"]
fn heavy_load() {
    run("json-server-heavy", 96, 40);
}
