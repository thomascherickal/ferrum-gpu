use super::*;

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
            .contains("handoff.md"),
        "unless the age limit is raised"
    );
    let _ = std::fs::write(d.join(".superpowers/handoff-fired"), "");
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

/// Regression: a session started in a parent directory has `CLAUDE_PROJECT_DIR` pointing there,
/// where `scripts/` does not exist; the hook commands ran `bash <missing file>` and exited 127
/// after every tool call. Every command in `.claude/settings.json` must succeed from anywhere.
#[test]
fn the_settings_commands_succeed_when_the_project_dir_is_not_this_repository() {
    let text = std::fs::read_to_string(repo().join(".claude/settings.json")).unwrap();
    let commands: Vec<String> = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("\"command\": "))
        .map(|c| {
            let c = c.trim_end_matches(',');
            // a JSON string: strip the quotes and undo the two escapes it uses
            c[1..c.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        })
        .collect();
    assert_eq!(commands.len(), 3, "status line and two hooks: {commands:?}");
    let elsewhere = std::env::temp_dir().join(format!("ferrum_hook_cwd_{}", std::process::id()));
    std::fs::create_dir_all(&elsewhere).unwrap();
    for cwd in [repo(), elsewhere.clone()] {
        for c in &commands {
            let mut child = Command::new("bash")
                .args(["-c", c])
                .current_dir(&cwd)
                .env("CLAUDE_PROJECT_DIR", &elsewhere)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            child.stdin.take().unwrap().write_all(b"{}").unwrap();
            let status = child.wait().unwrap();
            assert!(status.success(), "{c}\nin {}: {status}", cwd.display());
        }
    }
    std::fs::remove_dir_all(elsewhere).ok();
}
