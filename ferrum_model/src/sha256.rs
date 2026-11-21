//! SHA-256 (FIPS 180-4) in safe Rust, so token files can record which tokenizer file they were
//! made with without a dependency. Not constant-time; it hashes public files.

mod k;
pub use k::*;
mod sha256_hex;
pub use sha256_hex::*;

#[cfg(test)]
mod tests;
