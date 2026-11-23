use super::*;

/// Name the stage of a device operation whose error `r` carries: `"{what}: {error}"`.
pub(crate) fn stage<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> Result<T, GpuError> {
    r.map_err(|e| GpuError(format!("{what}: {e}")))
}
