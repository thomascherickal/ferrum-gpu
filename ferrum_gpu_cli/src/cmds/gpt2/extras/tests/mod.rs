//! Tests for the CPU-only training options: parsing, the device rule, and the engine they build.

mod cli;
mod device;
mod engine;
mod parse;

use crate::cmds::{table, DEVICE};
use crate::flags::{switch, val, Args};

/// Parse `s` as `train` would see these options.
fn args(s: &str) -> Args {
    let words: Vec<String> = s.split_whitespace().map(String::from).collect();
    Args::parse(
        &table(&[&DEVICE, &[switch("--tie"), val("--dropout")]]),
        &words,
    )
    .unwrap()
}
