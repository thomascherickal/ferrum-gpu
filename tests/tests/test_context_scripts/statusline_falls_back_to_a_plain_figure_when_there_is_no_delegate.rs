use super::*;

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
        out.contains("additionalContext") && out.contains("handoff.md") && out.contains("rule 12"),
        "{out}"
    );
    assert!(
        out.contains("\"hookEventName\":\"PostToolUse\"")
            || out.contains("\"hookEventName\": \"PostToolUse\""),
        "{out}"
    );
    assert!(latched(&d));
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
    assert!(!latched(&d));
    write_context(&d, 99.0, 0);
    assert!(run_hook(&d, &[], &[]).0.contains("handoff.md"));
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

/// The latch is a non-empty `handoff-fired` file; the hook empties it instead of deleting it.
fn latched(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join(".superpowers/handoff-fired")).is_ok_and(|s| !s.is_empty())
}
