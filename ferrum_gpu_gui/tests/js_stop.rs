//! Runs the Node tests of the Generate tab's stop-string helpers. Skips, with the reason printed,
//! when Node is not installed.

// the deadline wrapper (`timeout`) lives in the unix-only harness
#![cfg(unix)]

mod common;

use std::process::Command;

#[test]
fn the_stop_string_helpers_pass_their_node_tests() {
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("SKIP: node is not installed");
        return;
    }
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/js_tests/stop.test.mjs");
    let out = common::with_deadline("node", 60)
        .arg(script)
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.status.success(),
        "node tests failed:\n{stdout}\n{stderr}"
    );
    assert!(stdout.contains("OK "), "{stdout}");
}
