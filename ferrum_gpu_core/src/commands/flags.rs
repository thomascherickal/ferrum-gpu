//! A flag-table parser: each command lists the flags it accepts; anything else is an error.
use std::collections::HashMap;
use std::str::FromStr;
mod flag;
pub use flag::*;
mod args;

#[cfg(test)]
mod tests;
