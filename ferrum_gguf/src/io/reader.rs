//! GGUF v3 parser. Hostile-input safe: every read is bounds-checked, every size is checked
//! arithmetic, and nothing is allocated from a header number before it is proven to fit in
//! the file.
use crate::quant::{decode, nbytes};
use crate::{DEFAULT_ALIGNMENT, MAGIC};
use std::collections::HashMap;
mod gguferror;
pub use gguferror::*;
mod part;
mod part_2;

#[cfg(test)]
mod tests;
