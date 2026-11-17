//! Runs the Node tests of the Studio's page logic (shell, jobs client, paths, argument builders).
//! Skips, with the reason printed, when Node is not installed.

// the deadline wrapper (`timeout`) lives in the unix-only harness
#![cfg(unix)]

mod common;

use std::process::Command;

/// Run `js_tests/<name>` under node and return its stdout; `None` when node is missing.
fn node(name: &str) -> Option<String> {
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("SKIP: node is not installed");
        return None;
    }
    let script = format!("{}/js_tests/{name}", env!("CARGO_MANIFEST_DIR"));
    let out = common::with_deadline("node", 60)
        .arg(script)
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.status.success(),
        "node tests failed:\n{stdout}\n{stderr}"
    );
    Some(stdout)
}

#[test]
fn the_studio_shell_passes_its_node_tests() {
    if let Some(out) = node("shell.test.mjs") {
        assert!(out.contains("OK "), "{out}");
    }
}

#[test]
fn the_studio_page_logic_passes_its_node_tests() {
    if let Some(out) = node("studio.test.mjs") {
        assert!(out.contains("OK "), "{out}");
    }
}
