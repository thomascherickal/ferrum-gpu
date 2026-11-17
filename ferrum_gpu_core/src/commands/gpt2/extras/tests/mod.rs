//! Tests for the CPU-only training options: parsing, the device rule, and the engine they build.

mod cli;
mod device;
mod engine;
mod gpu;
mod parse;

use crate::commands::flags::{switch, val, Args};
use crate::commands::{table, DEVICE};

/// Parse `s` as `train` would see these options.
fn args(s: &str) -> Args {
    let words: Vec<String> = s.split_whitespace().map(String::from).collect();
    Args::parse(
        &table(&[&DEVICE, &[switch("--tie"), val("--dropout")]]),
        &words,
    )
    .unwrap()
}
