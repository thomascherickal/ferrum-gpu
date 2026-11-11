#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! The `ferrum-gpu` command line (Phase 6.1): a flag-table parser that rejects unknown flags, and
//! the commands `train`, `generate`, `eval`, `run-gguf`, `finetune-gguf` and `info`.

pub mod cmds;
pub mod device;
pub mod flags;
