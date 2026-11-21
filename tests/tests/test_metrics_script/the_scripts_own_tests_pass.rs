use super::*;

#[test]
fn the_scripts_own_tests_pass() {
    if !have_python() {
        return;
    }
    // metrics, rules, duplicates, timing and profile reports, next steps and the push gate
    let mut tests: Vec<_> = std::fs::read_dir(repo().join("scripts"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("test_") && n.ends_with(".py"))
        })
        .collect();
    tests.sort();
    assert!(tests.len() >= 7, "{tests:?}");
    for t in tests {
        let o = Command::new("python3").arg(&t).output().unwrap();
        assert!(
            o.status.success(),
            "{}: {}",
            t.display(),
            String::from_utf8_lossy(&o.stderr)
        );
    }
}

#[test]
fn the_pre_commit_hook_is_executable_and_never_blocks_a_commit() {
    use std::os::unix::fs::PermissionsExt;
    let hook = repo().join("scripts/githooks/pre-commit");
    let mode = std::fs::metadata(&hook).unwrap().permissions().mode();
    assert!(mode & 0o111 != 0, "the hook must be executable");
    let text = read("scripts/githooks/pre-commit");
    assert!(text.contains("metrics.py") && text.contains("--next-commit"));
    assert!(
        text.trim_end().ends_with("exit 0"),
        "the hook ends with exit 0"
    );
}

#[test]
fn coverage_and_the_gate_feed_metrics_md() {
    assert!(read("scripts/coverage.sh").contains("metrics.py --coverage"));
    assert!(read("scripts/gate.sh").contains("metrics.py --gate"));
    for hook in ["pre-commit", "pre-push"] {
        let text = read(&format!("scripts/githooks/{hook}"));
        assert!(
            text.contains("next_steps.py"),
            "{hook} refreshes next-steps.md"
        );
    }
    assert!(read("scripts/githooks/pre-push").contains("gate_ok.py"));
    // a merge keeps our side of metrics.md: the next commit refreshes it
    assert!(read(".gitattributes").contains("metrics.md merge=ours"));
}

#[test]
fn metrics_md_has_its_sections_and_markers() {
    let text = read("metrics.md");
    for needle in [
        "# Project metrics",
        "## Size",
        "## Tests",
        "## Structure",
        "## Dependencies",
        "## History",
        "## Gate",
        "## Test coverage",
        "<!-- gate:begin -->",
        "<!-- gate:end -->",
        "<!-- coverage:begin -->",
        "<!-- coverage:end -->",
    ] {
        assert!(text.contains(needle), "metrics.md lacks {needle:?}");
    }
}
