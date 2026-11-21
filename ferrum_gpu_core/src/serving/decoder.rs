//! The decoders the routes talk to: a stub for tests, a Llama-style byte model, and a GPT-2
//! model with its BPE tokenizer.
use crate::commands::gpt2::gen::{run, Loaded, Opts};
use crate::engine::{Backend, GenOpts};
use crate::tokenizer::{decode as bytes_to_text, encode as text_to_bytes};
use ferrum_model::sampling::Params;
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::llama::Llama;
use ferrum_tok::ByteBpeTokenizer;
use std::sync::{Arc, Mutex};

/// Cut `text` before the first of `stops` (shared with the GUI).
pub use crate::stop::cut_at_stop;
mod info;
pub use info::*;
mod gpt2decoder;
pub use gpt2decoder::*;
mod stubdecoder;
pub use stubdecoder::*;

#[cfg(test)]
mod tests;
