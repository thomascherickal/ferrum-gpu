//! A real `ollama serve` for the tests: a private port, a private `HOME` and model directory (it
//! writes an identity key and its blobs there, never into the user's own), in its own process group.
use super::model::{fixture, NEW_TOKENS, PROMPT};
use super::proc::{wait_until, with_deadline, Group};
use ferrum_gpu_core::json::Json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
mod mode;
pub use mode::*;
mod free_port;
pub use free_port::*;
mod start;
pub use start::*;
