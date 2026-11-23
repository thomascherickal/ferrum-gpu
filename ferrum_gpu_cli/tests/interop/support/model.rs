//! The small models every interop test uses, trained once per test process.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
mod sentence;
pub(crate) use sentence::*;
mod fixture;
pub(crate) use fixture::*;
mod ours;
pub(crate) use ours::*;
