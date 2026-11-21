//! Running tools: under a deadline, and servers in their own process group that are ended without
//! ever signalling anything else.
//!
//! The group signals follow CLAUDE.md rule 16 (`kill -s TERM -- -<pgid>`, never a bare negative
//! number, never group 0 or 1). `ferrum_gpu_gui/tests/common` has the same logic for Chrome; this
//! is a second copy because the two live in different test crates.
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
mod with_deadline;
pub use with_deadline::*;
mod wait_until;
pub use wait_until::*;
