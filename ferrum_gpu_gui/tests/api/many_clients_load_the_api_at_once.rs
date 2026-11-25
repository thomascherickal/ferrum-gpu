//! Many clients at once against the GUI API on the CPU backend: every answer is a 200 with JSON,
//! none is slow, and the server still answers afterwards. The `LOAD` line goes to system-profile.md
//! (`scripts/load_profile.py`). The heavier run, on request:
//! `cargo test --release -p ferrum_gpu_gui --test api -- --ignored heavy_gui_load --nocapture`
use super::*;
use ferrum_test_support::load::{drive, summary};

fn run(name: &str, clients: usize, each: usize) {
    let (srv, _app) = serve(cpu_app());
    let addr = srv.addr;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/api");
    let paths = [
        "/api/info".to_string(),
        "/api/serve/status".to_string(),
        format!("/api/fs/stat?path={dir}"),
    ];
    let (answers, wall) = drive(addr, clients, each, |i| {
        let path = &paths[i % paths.len()];
        format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\n\r\n").into_bytes()
    });
    assert_eq!(answers.len(), clients * each);
    for a in &answers {
        assert_eq!(a.status, 200, "{}", a.body);
        assert!(Json::parse(&a.body).is_ok(), "{}", a.body);
        assert!(a.took < Duration::from_secs(60), "{:?}", a.took);
    }
    println!("{}", summary(name, clients, &answers, wall));
    assert_eq!(get(addr.port(), "/api/info").status, 200);
    srv.stop();
}

#[test]
fn many_clients_load_the_api_at_once() {
    run("gui-api", 32, 6);
}

#[test]
#[ignore = "heavy: 3,840 requests from 96 clients"]
fn heavy_gui_load() {
    run("gui-api-heavy", 96, 40);
}
