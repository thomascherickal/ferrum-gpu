use super::*;
use std::sync::OnceLock;

/// Every registered kernel composed with its fragments, in [`SOURCES`] order.
fn composed() -> &'static [Result<String, String>] {
    static COMPOSED: OnceLock<Vec<Result<String, String>>> = OnceLock::new();
    COMPOSED.get_or_init(|| SOURCES.iter().map(|(_, s)| compose(s, FRAGMENTS)).collect())
}

/// The composed source of kernel `path` (`group/name`), or why there is none.
pub fn kernel(path: &str) -> Result<&'static str, String> {
    let at = SOURCES
        .iter()
        .position(|(n, _)| *n == path)
        .ok_or_else(|| format!("no WGSL kernel `{path}`"))?;
    composed()[at].as_deref().map_err(Clone::clone)
}

/// The composed source of kernel `path` (`group/name`), if registered.
pub fn source(path: &str) -> Option<&'static str> {
    kernel(path).ok()
}
