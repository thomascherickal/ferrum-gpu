use super::*;
use crate::app::DeviceChoice;
use crate::engine::CpuBackend;
use std::sync::Arc;

fn app() -> Arc<App> {
    Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ))
}

fn call(a: &App, method: &str, path: &str, body: &[u8]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: vec![],
        headers: vec![],
        body: body.to_vec(),
    };
    let r = route(a, &req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

#[test]
fn only_the_serve_and_tools_paths_are_handled() {
    assert!(handles("/api/serve/status") && handles("/api/serve/start") && handles("/api/tools"));
    assert!(!handles("/api/serve") && !handles("/api/toolsx") && !handles("/api/jobs"));
}

#[test]
fn an_idle_slot_reports_not_running_and_stop_has_nothing_to_do() {
    let a = app();
    let (s, j) = call(&a, "GET", "/api/serve/status", b"");
    assert_eq!(s, 200);
    assert_eq!(field(&j, "running"), Json::Bool(false));
    assert_eq!(field(&j, "addr"), Json::Null);
    assert_eq!(
        call(&a, "POST", "/api/serve/stop", b"").1.get("stopped"),
        Some(&Json::Bool(false))
    );
}

#[test]
fn bad_start_requests_are_400s_that_say_why() {
    let a = app();
    for (body, needle) in [
        (&br#"{"args":"x"}"#[..], "list of strings"),
        (br#"{"args":[1]}"#, "list of strings"),
        (br#"{}"#, "list of strings"),
        (b"{nope", "bad JSON"),
        (&[0xff, 0xfe][..], "UTF-8"),
        (br#"{"args":["--model","a\u0000b"]}"#, "NUL"),
        (br#"{"args":["--port","0"]}"#, "--model"),
        (
            br#"{"args":["--model","/no/such/model.gguf","--port","0","--device","cpu"]}"#,
            "no/such/model.gguf",
        ),
        (br#"{"args":["--nope"]}"#, "--nope"),
    ] {
        let (s, j) = call(&a, "POST", "/api/serve/start", body);
        assert_eq!(s, 400, "{}: {j}", String::from_utf8_lossy(body));
        let msg = field(&j, "error");
        assert!(msg.as_str().unwrap().contains(needle), "{msg}");
    }
    // the failed attempts are in the status lines, and nothing is running
    let (_, st) = call(&a, "GET", "/api/serve/status", b"");
    assert_eq!(field(&st, "running"), Json::Bool(false));
}

#[test]
fn a_server_starts_through_the_api_serves_and_stops() {
    use std::io::{Read, Write};
    let dir = std::env::temp_dir().join(format!("ferrum_servingapi_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("c.txt"),
        "the quick brown fox jumps over the lazy dog. ".repeat(60),
    )
    .unwrap();
    let model = dir.join("m.gguf");
    let train: Vec<String> = format!(
        "train --preset tiny --steps 3 --batch 2 --vocab 300 --device cpu --data {} --out {}",
        dir.join("c.txt").display(),
        model.display()
    )
    .split_whitespace()
    .map(String::from)
    .collect();
    ferrum_gpu_core::commands::run(&train, &mut Vec::new()).unwrap();

    let a = app();
    let body = format!(
        r#"{{"args":["--model","{}","--port","0","--device","cpu","--threads","1"]}}"#,
        model.display()
    );
    let (s, j) = call(&a, "POST", "/api/serve/start", body.as_bytes());
    assert_eq!(s, 200, "{j}");
    assert_eq!(field(&j, "running"), Json::Bool(true));
    let addr = field(&j, "addr").as_str().unwrap().to_string();
    let mut c = std::net::TcpStream::connect(&addr).unwrap();
    write!(
        c,
        "GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    // a second start is refused while it runs
    let (s, j) = call(&a, "POST", "/api/serve/start", body.as_bytes());
    assert_eq!(s, 400);
    assert!(field(&j, "error")
        .as_str()
        .unwrap()
        .contains("already running"));
    let (_, st) = call(&a, "GET", "/api/serve/status", b"");
    assert_eq!(field(&st, "running"), Json::Bool(true));
    let Json::Arr(lines) = field(&st, "lines") else {
        panic!("{st}")
    };
    assert!(lines.iter().any(|l| l
        .as_str()
        .is_some_and(|t| t.contains("ferrum-gpu server: http://"))));
    assert_eq!(
        call(&a, "POST", "/api/serve/stop", b"").1.get("stopped"),
        Some(&Json::Bool(true))
    );
    assert_eq!(
        field(&call(&a, "GET", "/api/serve/status", b"").1, "running"),
        Json::Bool(false)
    );
}

#[test]
fn tools_are_listed_with_a_path_or_the_reason_they_are_missing() {
    let a = app();
    let (s, j) = call(&a, "GET", "/api/tools", b"");
    assert_eq!(s, 200);
    let Json::Arr(tools) = field(&j, "tools") else {
        panic!("{j}")
    };
    let names: Vec<_> = tools
        .iter()
        .map(|t| field(t, "name").as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["ollama", "lms"]);
    for t in &tools {
        // exactly one of path and note is set
        let (p, n) = (field(t, "path"), field(t, "note"));
        assert!(
            matches!(
                (&p, &n),
                (Json::Str(_), Json::Null) | (Json::Null, Json::Str(_))
            ),
            "{t}"
        );
        if let Json::Str(note) = n {
            assert!(
                note.contains("cannot find") || note.contains("no such file"),
                "{note}"
            );
        }
    }
}

#[test]
fn other_methods_and_unknown_routes_are_refused() {
    let a = app();
    assert_eq!(call(&a, "GET", "/api/serve/start", b"").0, 405);
    assert_eq!(call(&a, "POST", "/api/serve/status", b"").0, 405);
    assert_eq!(call(&a, "POST", "/api/tools", b"").0, 405);
    assert_eq!(call(&a, "GET", "/api/serve/nope", b"").0, 404);
}

#[test]
fn the_api_router_sends_these_paths_here() {
    let a = app();
    for p in ["/api/serve/status", "/api/tools"] {
        let req = Request {
            method: "GET".into(),
            path: p.into(),
            query: vec![],
            headers: vec![],
            body: vec![],
        };
        assert_eq!(crate::api::route(&a, &req).status, 200, "{p}");
    }
}
