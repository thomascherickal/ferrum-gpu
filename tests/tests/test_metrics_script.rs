#![forbid(unsafe_code)]
// bash and python3 scripts with unix file modes: not built on Windows
#![cfg(unix)]
//! `metrics.md` and the machinery that keeps it current: the generator's own unit tests, the
//! pre-commit hook, and the places that must feed it (`coverage.sh`, `gate.sh`). Without python3 the
//! generator tests skip with a printed reason.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn have_python() -> bool {
    let ok = Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("SKIP: python3 is not installed");
    }
    ok
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn the_generators_own_tests_pass() {
    if !have_python() {
        return;
    }
    let o = Command::new("python3")
        .arg(repo().join("scripts/test_metrics.py"))
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn the_pre_commit_hook_is_executable_and_never_blocks_a_commit() {
    use std::os::unix::fs::PermissionsExt;
    let hook = repo().join("scripts/githooks/pre-commit");
    let mode = std::fs::metadata(&hook).unwrap().permissions().mode();
    assert!(mode & 0o111 != 0, "the hook must be executable");
    let text = read("scripts/githooks/pre-commit");
    assert!(text.contains("metrics.py") && text.contains("--next-commit"));
    assert!(text.trim_end().ends_with("exit 0"), "the hook ends with exit 0");
}

#[test]
fn coverage_and_the_gate_feed_metrics_md() {
    assert!(read("scripts/coverage.sh").contains("metrics.py --coverage"));
    assert!(read("scripts/gate.sh").contains("metrics.py --gate"));
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
