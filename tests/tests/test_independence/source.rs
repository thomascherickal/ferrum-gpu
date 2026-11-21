use super::manifest::stays_inside;
use std::path::Path;

/// Source lines that use ferrum_core or read files outside the repo. A path literal in an
/// `include_str!`/`include_bytes!`/`read` call is resolved against the file's directory `dir`
/// and must stay under `root` (so `"../shaders/x.wgsl"` inside the repo is fine, while
/// `"../../../ferrum-cpu/x"` is not).
pub(super) fn source_violations(text: &str, dir: &Path, root: &Path) -> Vec<String> {
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
pub(super) fn quoted(line: &str) -> Option<String> {
    Some(line.split('"').nth(1)?.to_string())
}

pub(super) fn unsafe_lines(text: &str) -> Vec<String> {
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
