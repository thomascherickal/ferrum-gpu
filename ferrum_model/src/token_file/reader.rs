//! Streaming reader of a token file: the header is parsed, the ids are read window by window.
use super::{element_type, TENSOR, TOKEN_FILE_ARCH, VERSION};
use ferrum_gguf::{Gguf, Value};
use ferrum_ref::trainer::TokenSource;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;
mod is_token_file;
pub use is_token_file::*;
mod metadata;
mod open;
mod token_source;
