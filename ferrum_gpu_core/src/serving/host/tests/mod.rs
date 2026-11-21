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

mod a_failed_start_says_why_and_leaves_the_slot_free;
mod an_empty_slot_is_not_running_and_has_nothing_to_stop;
