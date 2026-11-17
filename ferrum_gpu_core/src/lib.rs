#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! The shared core of ferrum-gpu. Every front end (the `ferrum-gpu` command line, the
//! `ferrum-gpu-server` JSON API, the web GUI and the desktop Studio) sits on this crate and on nothing
//! that sits on another front end:
//!
//! - [`engine`]: the `Backend` trait with its CPU and GPU implementations, training and generation
//!   loops, validation split and loss.
//! - [`commands`]: every `ferrum-gpu` command (train, finetune, eval, generate, prepare, tokenize, the
//!   exports), driven by an argument list and writing its report to any `Write`; the CLI is a thin
//!   `main` over [`commands::run`], and the Studio runs the same functions in-process.
//! - [`session`]: choosing a device (`cpu`, `gpu`, `auto`) and holding the backend that was opened.
//! - [`cancel`]: one cooperative cancel flag (Ctrl-C in a terminal, Stop in a window).
//! - [`json`], [`http`], [`stop`], [`tokenizer`]: the std-only helpers the servers and the byte-level
//!   path use.

pub mod cancel;
pub mod commands;
pub mod engine;
pub mod http;
pub mod json;
pub mod session;
pub mod stop;
pub mod tokenizer;
