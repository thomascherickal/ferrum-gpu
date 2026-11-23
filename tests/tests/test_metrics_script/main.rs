#![forbid(unsafe_code)]
//! `metrics.md` and the machinery that keeps it current: the generator's own unit tests, the
//! pre-commit hook, and the places that must feed it (`coverage.sh`, `gate.sh`). Without Python the
//! generator tests skip with a printed reason.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The Python interpreter (`python3`, or `python` where Windows installs that name), if any.
fn python() -> Option<&'static str> {
    let found = ["python3", "python"].into_iter().find(|p| {
        Command::new(p)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    });
    if found.is_none() {
        eprintln!("SKIP: python is not installed");
    }
    found
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

mod the_old_shell_names_are_wrappers;
mod the_scripts_own_tests_pass;
