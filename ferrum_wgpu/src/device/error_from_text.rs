use super::GpuError;

/// A [`GpuError`] from any message; the one place the engine builds one from text.
pub(crate) fn err(message: impl Into<String>) -> GpuError {
    GpuError(message.into())
}
