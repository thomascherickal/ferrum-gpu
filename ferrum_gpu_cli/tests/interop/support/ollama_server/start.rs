use super::*;

/// Start a server for test `tag`; `None` if it never became ready. Its output goes to a log file.
pub(crate) fn start(bin: &Path, tag: &str, mode: Mode) -> Option<Server> {
    let home = fixture().dir.join(format!("ollama-{tag}"));
    std::fs::create_dir_all(home.join("models")).unwrap();
    let host = format!("127.0.0.1:{}", free_port());
    let log = home.join("server.log");
    let mut s = Server {
        _group: None,
        host: host.clone(),
        home,
        bin: bin.to_path_buf(),
        mode,
        log: log.clone(),
    };
    let mut cmd = with_deadline(bin, 900);
    cmd.arg("serve");
    for (k, v) in s.env() {
        cmd.env(k, v);
    }
    let out = std::fs::File::create(&log).expect("create the server log");
    let err = out.try_clone().expect("clone the log handle");
    s._group = Some(
        cmd.guard(Stdio::from(out), Stdio::from(err))
            .expect("start the server"),
    );
    let ready = wait_until(60, || {
        http(&host, "GET", "/api/version", "").is_some_and(|(c, _)| c == 200)
    });
    ready.then_some(s)
}

/// One HTTP request; the status and the (de-chunked) body, or `None` if the connection failed.
pub(crate) fn http(host: &str, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
    let mut c = TcpStream::connect(host).ok()?;
    c.set_read_timeout(Some(std::time::Duration::from_secs(300)))
        .ok()?;
    write!(
        c,
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut raw = Vec::new();
    c.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n")?;
    let status = head.split(' ').nth(1)?.parse().ok()?;
    let body = if head.to_lowercase().contains("transfer-encoding: chunked") {
        dechunk(body)
    } else {
        body.to_string()
    };
    Some((status, body))
}

pub(super) fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((len, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
        if n == 0 || rest.len() < n {
            break;
        }
        out.push_str(&rest[..n]);
        s = rest[n..].trim_start_matches("\r\n");
    }
    out
}
