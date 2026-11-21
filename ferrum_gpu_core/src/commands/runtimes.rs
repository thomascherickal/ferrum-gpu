//! `export-ollama` and `export-lmstudio`: hand a GPT-2 GGUF to Ollama or LM Studio. The Modelfile
//! and the `lms` command lines follow what ferrum-cpu uses, so a model made here is registered the
//! same way.
use crate::commands::flags::Args;
use crate::commands::say;
use crate::json::Json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
mod modelfile_m;
pub use modelfile_m::*;
mod export_lmstudio_m;
pub use export_lmstudio_m::*;

#[cfg(test)]
mod tests;
