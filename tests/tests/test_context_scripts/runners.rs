use super::fixtures::repo;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub(super) fn run_statusline(project: &Path, delegate: &Path, json: &str) -> (String, i32) {
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

pub(super) fn run_hook(project: &Path, args: &[&str], envs: &[(&str, &str)]) -> (String, i32) {
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
