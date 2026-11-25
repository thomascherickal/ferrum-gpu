//! The file API and the request reader against hostile input, over real TCP. The file API is a
//! read-only view of the user's own files (no root to escape, by design: `..`, absolute paths and
//! symlinks resolve like in a file manager), so what must hold is: NUL bytes and bad escapes are
//! refused, oversized targets are refused before any file-system call, a symlink loop ends at the
//! depth limit, and a client that sends slowly holds only its own connection.
use super::*;
use std::io::Write;
use std::net::TcpStream;

fn status(p: u16, target: &str) -> (u16, String) {
    let r = get(p, target);
    (r.status, r.text())
}

#[test]
fn hostile_paths_get_a_clear_status() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let here = env!("CARGO_MANIFEST_DIR");
    for dir in [format!("{here}/tests/.."), format!("{here}/tests/%2e%2e")] {
        let (s, body) = status(p, &format!("/api/fs/list?dir={dir}"));
        assert!(s == 200 && body.contains("\"Cargo.toml\""), "{dir}: {body}");
    }
    for (target, want, message) in [
        ("/api/fs/list?dir=a%00b", 400, "NUL"),
        ("/api/fs/stat?path=%00", 400, "path is required"),
        ("/api/fs/preview?path=x%00y", 400, "path is required"),
        ("/api/fs/stat?path=%zz", 400, "bad percent escape"),
        ("/api/fs/stat?path=%ff", 400, "not UTF-8"),
        ("/..%2fCargo.toml", 404, ""),
        ("/shaders/..%5cCargo.toml", 404, ""),
        ("/index.html%00", 404, ""),
    ] {
        let (s, body) = status(p, target);
        assert!(s == want && body.contains(message), "{target}: {s} {body}");
    }
    let deep = "a/".repeat(4000); // 8 KB: a long path is just missing
    let (s, body) = status(p, &format!("/api/fs/stat?path=/{deep}"));
    assert!(s == 200 && body.contains("missing"), "{body}");
    let huge = "a/".repeat(9000); // past the 16 KiB head limit: refused unread
    assert_eq!(status(p, &format!("/api/fs/stat?path=/{huge}")).0, 431);
    let d = std::env::temp_dir().join(format!("ferrum_loop_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    if ferrum_test_support::link(&d, &d.join("loop")) {
        let (s, body) = status(p, &format!("/api/fs/stat?path={}", d.display()));
        assert!(s == 200 && body.contains("\"truncated\":true"), "{body}");
    }
    srv.stop();
}

#[test]
fn slow_clients_hold_only_their_own_connection() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut slow: Vec<TcpStream> = (0..20)
        .map(|_| {
            let mut c = TcpStream::connect(("127.0.0.1", p)).unwrap();
            c.write_all(b"GET /api/info HTTP/1.1\r\nHo").unwrap();
            c
        })
        .collect();
    let t0 = Instant::now();
    assert_eq!(get(p, "/api/info").status, 200);
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    std::thread::sleep(Duration::from_millis(200)); // well inside the read timeout
    for c in &mut slow {
        write!(c, "st: 127.0.0.1:{p}\r\n\r\n").unwrap();
    }
    for c in slow {
        let mut out = String::new();
        std::io::Read::read_to_string(&mut &c, &mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 200"), "{out:.80}");
    }
    let chunked = format!("POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
    assert_eq!(raw(p, chunked.as_bytes()).status, 501);
    srv.stop();
}
