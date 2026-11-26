use super::*;

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
        model.display().to_string().replace('\\', "\\\\") // JSON-escape Windows separators
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
