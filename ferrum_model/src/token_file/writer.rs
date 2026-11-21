//! One-pass atomic writer of a token file.
use super::{element_type, TokenFileMeta, TokenFileStats, TENSOR, TOKEN_FILE_ARCH, VERSION};
use ferrum_gguf::{GgufBuilder, Value, GGML_I32};
use std::io::{BufWriter, Seek, SeekFrom, Write};
mod header_bytes;
pub use header_bytes::*;
mod write;
pub use write::*;
