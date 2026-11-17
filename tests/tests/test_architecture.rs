//! The dependency direction of the crates, checked from the manifests: the shared core sits under
//! every front end, and no front end sits under another one.
//!
//! ```text
//! ferrum_gpu_cli, ferrum_gpu_gui, ferrum_gpu_server  ->  ferrum_gpu_core  ->  engine crates
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

const FRONT_ENDS: [&str; 3] = ["ferrum_gpu_cli", "ferrum_gpu_gui", "ferrum_gpu_server"];

#[test]
fn the_core_depends_on_no_front_end() {
    let d = deps("ferrum_gpu_core");
    assert!(!d.is_empty(), "the core has engine dependencies");
    for f in FRONT_ENDS {
        assert!(
            !d.contains(&f.to_string()),
            "ferrum_gpu_core must not depend on {f}"
        );
    }
}

#[test]
fn the_cli_and_the_gui_each_depend_on_the_core_and_not_on_each_other() {
    for (krate, other) in [
        ("ferrum_gpu_cli", "ferrum_gpu_gui"),
        ("ferrum_gpu_gui", "ferrum_gpu_cli"),
    ] {
        let d = deps(krate);
        assert!(
            d.contains(&"ferrum_gpu_core".to_string()),
            "{krate} uses the core"
        );
        assert!(
            !d.contains(&other.to_string()),
            "{krate} must not depend on {other}"
        );
    }
}

#[test]
fn the_server_depends_on_the_core_and_the_cli_but_not_the_gui() {
    let d = deps("ferrum_gpu_server");
    assert!(d.contains(&"ferrum_gpu_core".to_string()));
    assert!(!d.contains(&"ferrum_gpu_gui".to_string()));
}

#[test]
fn the_manifest_reader_sees_only_path_dependencies_in_the_right_table() {
    let m = "[package]\nname = \"x\"\n[dependencies]\na = { path = \"../a\" }\nserde = \"1\"\nb = { path = \"../b\" }\n[dev-dependencies]\nc = { path = \"../c\" }\n";
    assert_eq!(names(m), ["a", "b"]);
    assert!(names("[package]\nname = \"x\"\n").is_empty());
}
