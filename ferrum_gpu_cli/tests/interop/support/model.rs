//! The small models every interop test uses, trained once per test process.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
mod sentence;
pub use sentence::*;
mod fixture;
pub use fixture::*;
mod ours;
pub use ours::*;
