//! A slot that holds at most one running server, for a front end that starts and stops it from a
//! window (the Studio's Serve tab). It starts the server exactly as the command line does
//! ([`boot`](super::boot::boot)) and keeps the status lines it printed.
use super::boot::{boot, Booted};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;
mod max_lines;
pub use max_lines::*;
mod serve_slot;

#[cfg(test)]
mod tests;
