#![forbid(unsafe_code)]
// bash scripts with unix file modes: not built on Windows (the independence guard still is)
#![cfg(unix)]
//! Tests for `scripts/statusline.sh` and `scripts/handoff_check.sh`, the machinery behind
//! CLAUDE.md rule 12 (write handoff.md at 98% context). They run the real scripts under bash with
//! jq, a stub delegate and temp directories; without jq they skip with a printed reason.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn have_jq() -> bool {
    let ok = Command::new("jq")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("SKIP: jq is not installed");
    }
    ok
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gpu_ctx_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A delegate that proves it received the JSON: prints "DELEGATE:<input length>".
fn stub_delegate(dir: &Path) -> PathBuf {
    let p = dir.join("delegate.sh");
    std::fs::write(
        &p,
        "#!/usr/bin/env bash\ninput=$(cat)\nprintf 'DELEGATE:%s' \"${#input}\"\n",
    )
    .unwrap();
    p
}

fn run_statusline(project: &Path, delegate: &Path, json: &str) -> (String, i32) {
    let mut child = Command::new("bash")
        .arg(repo().join("scripts/statusline.sh"))
        .env("FERRUM_STATUSLINE_DELEGATE", delegate)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let doc = json.replace("PROJECT", &project.to_string_lossy());
    child
        .stdin
        .take()
        .unwrap()
        .write_all(doc.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn run_hook(project: &Path, args: &[&str], envs: &[(&str, &str)]) -> (String, i32) {
    let mut c = Command::new("bash");
    c.arg(repo().join("scripts/handoff_check.sh"))
        .args(args)
        .current_dir(project)
        .env("CLAUDE_PROJECT_DIR", project);
    for (k, v) in envs {
        c.env(k, v);
    }
    let out = c.output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn write_context(project: &Path, used: f64, age_secs: u64) {
    let dir = project.join(".superpowers");
    std::fs::create_dir_all(&dir).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    std::fs::write(
        dir.join("context.json"),
        format!(
            "{{\"context_used_percentage\": {used}, \"written_at\": {}}}",
            now - age_secs
        ),
    )
    .unwrap();
}

fn read_used(project: &Path) -> f64 {
    let text = std::fs::read_to_string(project.join(".superpowers/context.json")).unwrap();
    let v: f64 = text
        .split("\"context_used_percentage\":")
        .nth(1)
        .unwrap()
        .split([',', '}'])
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    v
}

mod statusline_falls_back_to_a_plain_figure_when_there_is_no_delegate;
mod statusline_records_used_context_from_remaining_and_delegates_the_display;
mod the_hook_ignores_a_stale_figure_and_garbage;
