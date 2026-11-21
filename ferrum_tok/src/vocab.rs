// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/gguf_io.rs (vocabulary arrays only) at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! GGUF vocabulary arrays and standalone tokenizer files.
use crate::error::{InferError, Result};
use crate::pretokenize::byte_to_unicode;
use crate::{ByteBpeTokenizer, BASE_VOCAB};
use std::collections::HashMap;
mod from_gguf;
mod gguf_export;

#[cfg(test)]
mod tests;
