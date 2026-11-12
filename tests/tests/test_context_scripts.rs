#![forbid(unsafe_code)]
// bash scripts with unix file modes: not built on Windows (the independence guard still is)
#![cfg(unix)]
//! Tests for `scripts/statusline.sh` and `scripts/handoff_check.sh`, the machinery behind
//! CLAUDE.md rule 12 (write HANDOFF.md at 98% context). They run the real scripts under bash with
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

#[test]
fn statusline_records_used_context_from_remaining_and_delegates_the_display() {
    if !have_jq() {
        return;
    }
    let d = tmp("sl1");
    let delegate = stub_delegate(&d);
    let json = r#"{"cwd":"PROJECT","context_window":{"remaining_percentage":37.5}}"#;
    let (out, code) = run_statusline(&d, &delegate, json);
    assert_eq!(code, 0);
    assert!(out.starts_with("DELEGATE:"), "{out}");
    assert!((read_used(&d) - 62.5).abs() < 1e-9);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn statusline_prefers_the_used_figure_and_the_project_dir() {
    if !have_jq() {
        return;
    }
    let d = tmp("sl2");
    let project = d.join("proj");
    std::fs::create_dir_all(&project).unwrap();
    let delegate = stub_delegate(&d);
    let json = r#"{"cwd":"/nonexistent/elsewhere","workspace":{"project_dir":"PROJECT"},"context_window":{"used_percentage":91,"remaining_percentage":50}}"#;
    run_statusline(&project, &delegate, json);
    assert_eq!(
        read_used(&project),
        91.0,
        "used_percentage wins over remaining_percentage"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn statusline_records_the_figure_in_both_the_project_dir_and_the_cwd_when_they_differ() {
    if !have_jq() {
        return;
    }
    let d = tmp("sl2b");
    let (project, work) = (d.join("main"), d.join("worktree"));
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let delegate = stub_delegate(&d);
    let json = format!(
        r#"{{"cwd":"{}","workspace":{{"project_dir":"{}"}},"context_window":{{"used_percentage":55}}}}"#,
        work.display(),
        project.display()
    );
    run_statusline(&d, &delegate, &json);
    assert_eq!(read_used(&project), 55.0);
    assert_eq!(
        read_used(&work),
        55.0,
        "a worktree session's hook reads from its own directory"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn statusline_without_a_figure_writes_nothing_and_still_prints() {
    if !have_jq() {
        return;
    }
    let d = tmp("sl3");
    let delegate = stub_delegate(&d);
    let (out, code) = run_statusline(&d, &delegate, r#"{"cwd":"PROJECT"}"#);
    assert_eq!(code, 0);
    assert!(out.starts_with("DELEGATE:"));
    assert!(!d.join(".superpowers/context.json").exists());
    // Garbage on stdin must not break the UI either.
    let (out, code) = run_statusline(&d, &delegate, "not json at all {{{");
    assert_eq!(code, 0);
    assert!(out.starts_with("DELEGATE:"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn statusline_falls_back_to_a_plain_figure_when_there_is_no_delegate() {
    if !have_jq() {
        return;
    }
    let d = tmp("sl4");
    let (out, code) = run_statusline(
        &d,
        &d.join("missing.sh"),
        r#"{"cwd":"PROJECT","context_window":{"used_percentage":12}}"#,
    );
    assert_eq!((out.as_str(), code), ("ctx 12%", 0));
    let (out, _) = run_statusline(&d, &d.join("missing.sh"), r#"{"cwd":"PROJECT"}"#);
    assert_eq!(out, "ctx ?%");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_hook_is_silent_below_the_threshold_and_when_there_is_no_figure() {
    if !have_jq() {
        return;
    }
    let d = tmp("hk1");
    assert_eq!(
        run_hook(&d, &[], &[]),
        (String::new(), 0),
        "no figure at all"
    );
    write_context(&d, 50.0, 0);
    assert_eq!(run_hook(&d, &[], &[]), (String::new(), 0));
    write_context(&d, 97.9, 0);
    assert_eq!(run_hook(&d, &[], &[]), (String::new(), 0), "just under 98");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_hook_fires_once_at_98_then_again_only_after_usage_falls() {
    if !have_jq() {
        return;
    }
    let d = tmp("hk2");
    write_context(&d, 98.0, 0);
    let (out, code) = run_hook(&d, &[], &[]);
    assert_eq!(code, 0);
    assert!(
        out.contains("additionalContext") && out.contains("HANDOFF.md") && out.contains("rule 12"),
        "{out}"
    );
    assert!(
        out.contains("\"hookEventName\":\"PostToolUse\"")
            || out.contains("\"hookEventName\": \"PostToolUse\""),
        "{out}"
    );
    assert!(d.join(".superpowers/handoff-fired").exists());
    assert_eq!(run_hook(&d, &[], &[]).0, "", "does not nag a second time");
    write_context(&d, 99.5, 0);
    assert_eq!(
        run_hook(&d, &[], &[]).0,
        "",
        "still latched while usage stays high"
    );
    // Usage falls (a new session or a compaction): the latch resets and it can fire again.
    write_context(&d, 40.0, 0);
    assert_eq!(run_hook(&d, &[], &[]).0, "");
    assert!(!d.join(".superpowers/handoff-fired").exists());
    write_context(&d, 99.0, 0);
    assert!(run_hook(&d, &[], &[]).0.contains("HANDOFF.md"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_hook_names_the_event_it_runs_under_and_honours_the_overrides() {
    if !have_jq() {
        return;
    }
    let d = tmp("hk3");
    write_context(&d, 80.0, 0);
    let (out, _) = run_hook(&d, &["UserPromptSubmit"], &[("HANDOFF_AT", "75")]);
    assert!(out.contains("UserPromptSubmit"), "{out}");
    assert!(
        out.contains("75%"),
        "the message names the configured threshold: {out}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_hook_ignores_a_stale_figure_and_garbage() {
    if !have_jq() {
        return;
    }
    let d = tmp("hk4");
    write_context(&d, 99.0, 3600); // an hour old
    assert_eq!(run_hook(&d, &[], &[]), (String::new(), 0), "stale");
    assert!(
        run_hook(&d, &[], &[("HANDOFF_MAX_AGE", "7200")])
            .0
            .contains("HANDOFF.md"),
        "unless the age limit is raised"
    );
    let _ = std::fs::remove_file(d.join(".superpowers/handoff-fired"));
    std::fs::write(d.join(".superpowers/context.json"), "{{ not json").unwrap();
    assert_eq!(run_hook(&d, &[], &[]), (String::new(), 0), "garbage");
    std::fs::write(
        d.join(".superpowers/context.json"),
        r#"{"context_used_percentage": "lots", "written_at": 1}"#,
    )
    .unwrap();
    assert_eq!(run_hook(&d, &[], &[]), (String::new(), 0), "non-numeric");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_scripts_are_executable_and_documented() {
    for s in ["scripts/statusline.sh", "scripts/handoff_check.sh"] {
        let p = repo().join(s);
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with("#!/usr/bin/env bash"), "{s}");
        assert!(
            text.contains("rule 12"),
            "{s} should name the rule it implements"
        );
        use std::os::unix::fs::PermissionsExt;
        assert!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o111 != 0,
            "{s} must be executable"
        );
    }
}
