//! The pre-tokenized corpus file (`general.architecture = "ferrum-tokens"`), the same layout
//! ferrum-cpu writes: a GGUF with one `tokens` tensor of ids (I16 when the vocabulary fits, else
//! I32) and `ferrum.tokens.*` metadata. Training reads windows from disk, so a corpus larger
//! than memory can be trained on.

mod reader;
mod writer;

pub use reader::{is_token_file, TokenFile};
pub use writer::{header_bytes, write, TokenSink};

use ferrum_gguf::{GGML_I16, GGML_I32};

/// `general.architecture` of a token file.
pub const TOKEN_FILE_ARCH: &str = "ferrum-tokens";
/// Layout version under `ferrum.tokens.version`.
pub const VERSION: u32 = 1;
/// Name of the single tensor.
pub const TENSOR: &str = "tokens";

/// What a token file records about its ids (everything but the ids themselves).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenFileMeta {
    /// Ids are in `0..vocab_size`.
    pub vocab_size: usize,
    /// The id that separates documents, when the tokenizer has one.
    pub eos_id: Option<u32>,
    /// Lowercase hex SHA-256 of the tokenizer file the ids came from.
    pub tokenizer_sha256: String,
}

/// What a finished token file holds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenFileStats {
    /// Ids written.
    pub tokens: u64,
    /// Documents written (one more than the number of end-of-text ids; 0 when empty).
    pub documents: u64,
    /// Size of the file in bytes.
    pub bytes: u64,
}

/// The tensor type and bytes per id for a vocabulary: I16 holds ids up to 32767.
pub fn element_type(vocab_size: usize) -> (u32, usize) {
    if vocab_size <= i16::MAX as usize + 1 {
        (GGML_I16, 2)
    } else {
        (GGML_I32, 4)
    }
}

#[cfg(test)]
mod tests;
