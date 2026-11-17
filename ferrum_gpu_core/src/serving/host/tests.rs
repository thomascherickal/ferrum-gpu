use super::*;
use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

/// A tiny GPT-2 model trained through the real command, in a fresh temp folder.
fn model() -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_host_{}_{}", std::process::id(), nanos()));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("c.txt"),
        "the quick brown fox jumps over the lazy dog. ".repeat(60),
    )
    .unwrap();
    let out = d.join("m.gguf");
    let cmd = args(&format!(
        "train --preset tiny --steps 3 --batch 2 --vocab 300 --device cpu --data {} --out {}",
        d.join("c.txt").display(),
        out.display()
    ));
    crate::commands::run(&cmd, &mut Vec::new()).unwrap();
    out
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

fn get(addr: &str, path: &str, token: Option<&str>) -> String {
    let mut c = TcpStream::connect(addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    write!(
        c,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}Connection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn an_empty_slot_is_not_running_and_has_nothing_to_stop() {
    let s = ServeSlot::new();
    let st = s.status();
    assert!(!st.running && st.addr.is_none() && st.seconds.is_none() && st.lines.is_empty());
    assert!(!st.has_token);
    assert!(!s.stop());
}

#[test]
fn a_server_starts_answers_and_stops_and_a_second_start_is_refused_meanwhile() {
    let m = model();
    let slot = ServeSlot::new();
    let st = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu --threads 1",
            m.display()
        )))
        .unwrap();
    assert!(st.running && !st.has_token);
    let addr = st.addr.clone().unwrap();
    assert!(
        st.lines.iter().any(|l| l.starts_with("device: ")),
        "{:?}",
        st.lines
    );
    assert!(
        st.lines
            .iter()
            .any(|l| l.contains("ferrum-gpu server: http://")),
        "{:?}",
        st.lines
    );
    // it really serves
    let health = get(&addr, "/health", None);
    assert!(
        health.starts_with("HTTP/1.1 200") && health.contains("\"ok\""),
        "{health}"
    );
    let info = get(&addr, "/v1/info", None);
    assert!(info.contains("\"arch\"") || info.contains("gpt2"), "{info}");
    // one at a time
    let e = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu",
            m.display()
        )))
        .unwrap_err();
    assert!(e.contains("already running"), "{e}");
    assert!(slot.status().seconds.unwrap() >= 0.0);
    // stopping ends it
    assert!(slot.stop());
    let after = slot.status();
    assert!(!after.running && after.lines.last().unwrap() == "server stopped");
    assert!(
        TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(500)).is_err()
    );
    assert!(!slot.stop(), "a second stop does nothing");
}

#[test]
fn a_token_protects_the_server_and_is_reported() {
    let m = model();
    let slot = ServeSlot::new();
    let st = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu --token s3cret",
            m.display()
        )))
        .unwrap();
    assert!(st.has_token);
    assert!(st.lines.iter().any(|l| l.starts_with("auth:")));
    let addr = st.addr.unwrap();
    assert!(get(&addr, "/v1/info", None).starts_with("HTTP/1.1 401"));
    assert!(get(&addr, "/v1/info", Some("s3cret")).starts_with("HTTP/1.1 200"));
    assert!(
        get(&addr, "/health", None).starts_with("HTTP/1.1 200"),
        "health is open"
    );
    slot.stop();
}

#[test]
fn a_failed_start_says_why_and_leaves_the_slot_free() {
    let slot = ServeSlot::new();
    for (bad, needle) in [
        ("--port 0".to_string(), "--model"),
        (
            "--model /no/such.gguf --port 0 --device cpu".to_string(),
            "no/such.gguf",
        ),
        ("--model x --device tpu".to_string(), "tpu"),
        ("--model x --nope".to_string(), "--nope"),
        (
            "--model x --host 0.0.0.0".to_string(),
            "not a loopback address",
        ),
    ] {
        let e = slot.start(&args(&bad)).unwrap_err();
        assert!(e.contains(needle), "{bad}: {e}");
        let st = slot.status();
        assert!(!st.running);
        assert!(
            st.lines.last().unwrap().starts_with("could not start: "),
            "{:?}",
            st.lines
        );
    }
    // and a good start still works afterwards
    let m = model();
    slot.start(&args(&format!(
        "--model {} --port 0 --device cpu",
        m.display()
    )))
    .unwrap();
    slot.stop();
}

#[test]
fn the_kept_lines_are_bounded() {
    let lines = Arc::new(Mutex::new(vec![]));
    let mut w = Lines(lines.clone(), vec![]);
    for i in 0..(MAX_LINES + 30) {
        writeln!(w, "line {i}").unwrap();
    }
    write!(w, "unfinished").unwrap();
    w.flush().unwrap();
    let v = lock(&lines);
    assert_eq!(v.len(), MAX_LINES);
    assert_eq!(v.last().unwrap(), &format!("line {}", MAX_LINES + 29));
    assert_eq!(v[0], "line 30", "the oldest were dropped");
}
