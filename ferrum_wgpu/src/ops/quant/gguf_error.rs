use super::*;

impl From<ferrum_gguf::GgufError> for GpuError {
    fn from(e: ferrum_gguf::GgufError) -> Self {
        err(e.to_string())
    }
}
