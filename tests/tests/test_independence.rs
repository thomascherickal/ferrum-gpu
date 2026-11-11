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
fn manifest_violations(text: &str, dir: &Path, root: &Path) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter(|l| {
            l.contains("git =")
                || l.contains("ferrum_core")
                || l.contains("ferrum-cpu")
                || path_value(l).is_some_and(|p| !stays_inside(dir, &p, root))
        })
        .map(String::from)
        .collect()
}

/// The quoted value of `path = "..."` on a manifest line, if any.
fn path_value(line: &str) -> Option<String> {
    let rest = line.split("path").nth(1)?.trim_start().strip_prefix('=')?;
    let rest = rest.trim_start().strip_prefix('"')?;
    Some(rest.split('"').next()?.to_string())
}

/// True when `dir/rel`, with `..` resolved lexically, is still under `root`.
fn stays_inside(dir: &Path, rel: &str, root: &Path) -> bool {
    let mut parts: Vec<std::ffi::OsString> =
        dir.components().map(|c| c.as_os_str().to_owned()).collect();
    for seg in Path::new(rel).components() {
        match seg {
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::CurDir => {}
            other => parts.push(other.as_os_str().to_owned()),
        }
    }
    let resolved: PathBuf = parts.iter().collect();
    resolved.starts_with(root)
}

/// Source lines that use ferrum_core or read files outside the repo. A path literal in an
/// `include_str!`/`include_bytes!`/`read` call is resolved against the file's directory `dir`
/// and must stay under `root` (so `"../shaders/x.wgsl"` inside the repo is fine, while
/// `"../../../ferrum-cpu/x"` is not).
fn source_violations(text: &str, dir: &Path, root: &Path) -> Vec<String> {
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
                    && (l.contains("ferrum-cpu")
                        || quoted(l).is_some_and(|p| !stays_inside(dir, &p, root))))
        })
        .map(String::from)
        .collect()
}

/// The first double-quoted string on a line.
fn quoted(line: &str) -> Option<String> {
    Some(line.split('"').nth(1)?.to_string())
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

fn scan(ext: &str, name: Option<&str>, f: &dyn Fn(&str, &Path) -> Vec<String>) -> Vec<String> {
    let mut bad = vec![];
    for p in files(ext, name) {
        if p.ends_with("test_independence.rs") {
            continue;
        }
        for v in f(&fs::read_to_string(&p).unwrap(), p.parent().unwrap()) {
            bad.push(format!("{}: {v}", p.display()));
        }
    }
    bad
}

#[test]
fn no_manifest_points_at_ferrum_cpu_or_outside() {
    let root = root();
    let mut bad = vec![];
    for m in files("", Some("Cargo.toml")) {
        let text = fs::read_to_string(&m).unwrap();
        for v in manifest_violations(&text, m.parent().unwrap(), &root) {
            bad.push(format!("{}: {v}", m.display()));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_source_reaches_ferrum_cpu() {
    let bad = scan("rs", None, &|t, d| source_violations(t, d, &root()));
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn no_unsafe_anywhere() {
    let bad = scan("rs", None, &|t, _| unsafe_lines(t));
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn the_detectors_catch_what_they_should() {
    let (root, dir) = (Path::new("/r"), Path::new("/r/crate"));
    let mv = |t: &str| manifest_violations(t, dir, root);
    assert!(!mv("ferrum_core = { path = \"../ferrum-cpu/ferrum_core\" }").is_empty());
    assert!(!mv("x = { path = \"../../other\" }").is_empty());
    assert!(!mv("x = { git = \"https://e/x\" }").is_empty());
    assert!(mv("x = { path = \"../sibling\" }").is_empty());
    assert!(mv("x = { path = \"./inner\" }").is_empty());
    assert!(mv("# ferrum-cpu mention\nwgpu = \"30\"").is_empty());
    assert_eq!(
        path_value("x = { path = \"../a\", version = \"1\" }").as_deref(),
        Some("../a")
    );
    assert_eq!(path_value("wgpu = \"30\""), None);
    let sv = |t: &str| source_violations(t, dir, root);
    assert!(!sv("use ferrum_core::Tensor;").is_empty());
    assert!(!sv("let s = include_str!(\"../../ferrum-cpu/x\");").is_empty());
    assert!(
        !sv("let s = include_str!(\"../../other/x\");").is_empty(),
        "leaves the repo"
    );
    assert!(sv("// from ferrum_core::ops (provenance)").is_empty());
    assert!(sv("let s = include_str!(\"shaders/a.wgsl\");").is_empty());
    assert!(
        sv("let s = include_str!(\"../shaders/a.wgsl\");").is_empty(),
        "inside the repo"
    );
    assert!(!unsafe_lines("unsafe { x() }").is_empty());
    assert!(unsafe_lines("#![forbid(unsafe_code)]").is_empty());
}
