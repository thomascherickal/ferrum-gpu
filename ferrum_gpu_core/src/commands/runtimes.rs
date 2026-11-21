//! `export-ollama` and `export-lmstudio`: hand a GPT-2 GGUF to Ollama or LM Studio. The Modelfile
//! and the `lms` command lines follow what ferrum-cpu uses, so a model made here is registered the
//! same way.
use crate::commands::flags::Args;
use crate::commands::say;
use crate::json::Json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
mod modelfile;
pub use modelfile::*;
mod find_tool_in;
pub(crate) use find_tool_in::*;
mod export_ollama;
pub use export_ollama::*;
mod export_lmstudio;
pub use export_lmstudio::*;

#[cfg(test)]
mod tests;
