//! Runs every Node test of the Studio's page logic under `js_tests/` (shell, jobs client, paths, argument builders).
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

/// Test files other tests run themselves (they need a server or have their own harness).
const RUN_ELSEWHERE: [&str; 2] = ["reader.test.mjs", "stop.test.mjs"];

#[test]
fn every_node_test_file_passes() {
    let dir = format!("{}/js_tests", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".test.mjs") && !RUN_ELSEWHERE.contains(&n.as_str()))
        .collect();
    names.sort();
    assert!(names.len() >= 12, "found only {names:?}");
    for name in names {
        if let Some(out) = node(&name) {
            assert!(out.contains("OK "), "{name}: {out}");
        }
    }
}
