use std::path::{Path, PathBuf};

/// Manifest lines that would couple us to another checkout.
pub(super) fn manifest_violations(text: &str, dir: &Path, root: &Path) -> Vec<String> {
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
pub(super) fn path_value(line: &str) -> Option<String> {
    let rest = line.split("path").nth(1)?.trim_start().strip_prefix('=')?;
    let rest = rest.trim_start().strip_prefix('"')?;
    Some(rest.split('"').next()?.to_string())
}

/// True when `dir/rel`, with `..` resolved lexically, is still under `root`.
pub(super) fn stays_inside(dir: &Path, rel: &str, root: &Path) -> bool {
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
