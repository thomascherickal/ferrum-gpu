//! Tokenizer errors. Small on purpose: three failure kinds, all carrying a message.

use std::fmt;

/// Why a tokenizer operation failed.
#[derive(Debug, Clone, PartialEq)]
pub enum InferError {
    /// Sizes or lengths that must agree do not.
    DimMismatch(String),
    /// Data is not in the expected format.
    Format(String),
    /// A number could not be parsed.
    ParseError(String),
    /// Reading a corpus file failed.
    Io(String),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::DimMismatch(m)
            | InferError::Format(m)
            | InferError::ParseError(m)
            | InferError::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for InferError {}

impl From<std::io::Error> for InferError {
    fn from(e: std::io::Error) -> Self {
        InferError::Io(e.to_string())
    }
}

/// Result with [`InferError`].
pub type Result<T> = std::result::Result<T, InferError>;

/// Fail if `n` exceeds `max`; guards sizes read from untrusted data.
pub(crate) fn bounded(n: usize, max: usize, what: &str) -> Result<usize> {
    if n > max {
        return Err(InferError::Format(format!(
            "{what} = {n} exceeds what the data can hold ({max})"
        )));
    }
    Ok(n)
}
