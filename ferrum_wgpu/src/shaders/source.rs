use super::*;

/// The source of kernel `path` (`group/name`), if registered.
pub fn source(path: &str) -> Option<&'static str> {
    SOURCES.iter().find(|(n, _)| *n == path).map(|(_, s)| *s)
}
