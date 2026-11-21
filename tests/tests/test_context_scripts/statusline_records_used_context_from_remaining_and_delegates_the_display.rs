use super::*;

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
