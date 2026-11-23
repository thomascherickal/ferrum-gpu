//! Error paths of the GPT-2 commands that the happy-path tests never reach: mismatched inputs,
//! diverging runs, unwritable outputs. Each test uses a few hundred bytes of text and a model of
//! about ten thousand parameters.

use crate::commands::run;
use std::io::Write;
use std::path::{Path, PathBuf};
mod a_helper_command_that_fails_panics_naming_it;
mod fast;
mod generate;
pub(crate) mod sweep;
mod train;
mod unusable_inputs;
mod unusable_models;
mod unwritable_outputs;
pub(crate) use fast::*;
mod fail_nth;
