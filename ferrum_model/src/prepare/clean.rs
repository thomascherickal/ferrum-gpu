//! Per-document cleaning for `prepare`.
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/dataset.rs` (commit 56a9d3a), reduced to the
//! options `prepare` actually uses (whitespace collapsing, punctuation folding and control-character
//! stripping always on; Gutenberg stripping and lowercasing selectable).

mod is_invisible;
pub use is_invisible::*;
mod squeeze_spaces;
pub use squeeze_spaces::*;

#[cfg(test)]
mod tests;
