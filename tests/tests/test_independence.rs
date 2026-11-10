#![forbid(unsafe_code)]
//! Guards CLAUDE.md's independence rule and the no-unsafe rule.
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name == "target" || name == ".git" || name == ".claude" {
            continue;
        }
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn files(ext: &str, name: Option<&str>) -> Vec<PathBuf> {
    let mut all = vec![];
    walk(&root(), &mut all);
    all.into_iter()
        .filter(|p| match name {
            Some(n) => p.file_name().is_some_and(|f| f == n),
            None => p.extension().is_some_and(|e| e == ext),
        })
        .collect()
}

/// Manifest lines that would couple us to another checkout.
fn manifest_violations(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter(|l| {
            (l.contains("path") && l.contains('=') && l.contains(".."))
                || l.contains("git =")
                || l.contains("ferrum_core")
                || l.contains("ferrum-cpu")
        })
        .map(String::from)
        .collect()
}

/// Source lines that use ferrum_core or read files outside the repo.
fn source_violations(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//"))
        .filter(|l| {
            l.contains("ferrum_core::")
                || l.contains("use ferrum_core")
                || ((l.contains("include_str!")
                    || l.contains("include_bytes!")
                    || l.contains("read(\"")
                    || l.contains("read_to_string(\""))
                    && (l.contains("\"../") || l.contains("ferrum-cpu")))
        })
        .map(String::from)
        .collect()
}

fn unsafe_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//"))
        .filter(|l| {
            let l = l.replace("forbid(unsafe_code)", "");
            l.split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|w| w == "unsafe")
        })
        .map(String::from)
        .collect()
}

fn scan(ext: &str, name: Option<&str>, f: fn(&str) -> Vec<String>) -> Vec<String> {
    let mut bad = vec![];
    for p in files(ext, name) {
        if p.ends_with("test_independence.rs") {
            continue;
        }
        for v in f(&fs::read_to_string(&p).unwrap()) {
            bad.push(format!("{}: {v}", p.display()));
        }
    }
    bad
}

#[test]
fn no_manifest_points_at_ferrum_cpu_or_outside() {
    let bad = scan("", Some("Cargo.toml"), manifest_violations);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_source_reaches_ferrum_cpu() {
    let bad = scan("rs", None, source_violations);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_unsafe_anywhere() {
    let bad = scan("rs", None, unsafe_lines);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn the_detectors_catch_what_they_should() {
    assert!(
        !manifest_violations("ferrum_core = { path = \"../ferrum-cpu/ferrum_core\" }").is_empty()
    );
    assert!(!manifest_violations("x = { path = \"../other\" }").is_empty());
    assert!(!manifest_violations("x = { git = \"https://e/x\" }").is_empty());
    assert!(manifest_violations("# ferrum-cpu mention\nwgpu = \"30\"").is_empty());
    assert!(!source_violations("use ferrum_core::Tensor;").is_empty());
    assert!(!source_violations("let s = include_str!(\"../../ferrum-cpu/x\");").is_empty());
    assert!(source_violations("// from ferrum_core::ops (provenance)").is_empty());
    assert!(source_violations("let s = include_str!(\"shaders/a.wgsl\");").is_empty());
    assert!(!unsafe_lines("unsafe { x() }").is_empty());
    assert!(unsafe_lines("#![forbid(unsafe_code)]").is_empty());
}
