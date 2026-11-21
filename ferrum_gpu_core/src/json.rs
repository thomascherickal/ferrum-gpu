//! A small JSON reader and writer (std only). The reader is strict about syntax, bounds nesting
//! depth and refuses trailing data, because it parses request bodies from a browser.
use std::fmt::Write;
mod value;
pub use value::*;
mod part;
mod part_2;

#[cfg(test)]
mod tests;
