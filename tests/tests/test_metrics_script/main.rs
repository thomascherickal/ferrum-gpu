#![forbid(unsafe_code)]
// bash and python3 scripts with unix file modes: not built on Windows
#![cfg(unix)]
//! `metrics.md` and the machinery that keeps it current: the generator's own unit tests, the
//! pre-commit hook, and the places that must feed it (`coverage.sh`, `gate.sh`). Without python3 the
//! generator tests skip with a printed reason.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
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

mod the_scripts_own_tests_pass;
