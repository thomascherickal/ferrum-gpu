//! Looking inside a GGUF model without loading it: its header, metadata, tensors and size, read from
//! the start of the file only (`Gguf::parse_prefix`), so a model of any size can be inspected. The
//! reader is hostile-input safe (it is `ferrum_gguf`'s), and this module adds limits of its own.

use crate::json::Json;
use ferrum_gguf::{Gguf, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
pub mod kinds;
mod max_header;
pub mod scan;
pub mod tokenizer;
pub use max_header::*;
mod inspect_with;
pub use inspect_with::*;
mod to_json;
pub use to_json::*;

#[cfg(test)]
mod tests;
