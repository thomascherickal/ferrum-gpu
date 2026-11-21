//! The dependency direction of the crates, checked from the manifests: the shared core sits under
//! every front end, and no front end sits under another one.
//!
//! ```text
//! ferrum_gpu_cli, ferrum_gpu_gui, ferrum_gpu_server  ->  ferrum_gpu_core  ->  engine crates
//! (the commands live in the core; the CLI is a thin `main` over them)
//! ```
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The names of the path dependencies in a crate's `[dependencies]` table.
fn deps(krate: &str) -> Vec<String> {
    let text = fs::read_to_string(root().join(krate).join("Cargo.toml")).unwrap();
    names(&text)
}

fn names(manifest: &str) -> Vec<String> {
    let mut in_deps = false;
    let mut out = vec![];
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_deps = line == "[dependencies]";
            continue;
        }
        if in_deps && line.contains("path =") {
            if let Some((name, _)) = line.split_once('=') {
                out.push(name.trim().to_string());
            }
        }
    }
    out
}

const FRONT_ENDS: [&str; 4] = [
    "ferrum_gpu_cli",
    "ferrum_gpu_gui",
    "ferrum_gpu_server",
    "ferrum_gpu_studio",
];

mod the_core_depends_on_no_front_end;
mod the_manifest_reader_sees_only_path_dependencies_in_the_right_table;
