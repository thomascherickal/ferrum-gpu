//! Jobs: `ferrum-gpu` commands run **in this process**, on their own threads, with their output
//! streamed to whoever polls (the web page, the desktop window).
//!
//! A job runs exactly what the command line runs ([`crate::commands::run`]: an argument list in, a
//! report written to a `Write` out), so there is one implementation of every command. Output lines
//! are collected as they are written; progress lines are parsed into events ([`progress`]); a stop
//! raises the job's cancel flag, which the training and generation loops poll ([`crate::cancel`]).
//! A command that fails returns an error and a command that panics is caught: either ends the job,
//! never the application. What may run is fixed in [`allow`]; the JSON the page reads is in [`view`].

use progress::Event;
use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
pub mod allow;
mod max_lines;
pub mod progress;
pub mod runner;
pub mod view;
pub use max_lines::*;
mod job_poll;
mod job_state;
mod job_table;
pub use job_table::*;
mod registry;

#[cfg(test)]
mod tests;
