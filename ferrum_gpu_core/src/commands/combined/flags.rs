//! The flag tables of `run`'s two halves.

use crate::commands::flags::{switch, val, Flag};
use crate::commands::{gpt2, table, DEVICE, GEN, TRAIN};

/// What `train` accepts (including `--out`, which `run` refuses in favour of `--model`).
pub fn train_flags() -> Vec<Flag> {
    table(&[
        &DEVICE,
        &TRAIN,
        &gpt2::SIZE_FLAGS,
        &gpt2::TRAIN_FLAGS,
        &[val("--quant"), val("--arch"), val("--epochs")],
    ])
}

/// What `generate` accepts.
pub fn generate_flags() -> Vec<Flag> {
    table(&[
        &DEVICE,
        &GEN,
        &gpt2::GEN_FLAGS,
        &[val("--seed"), val("--max-new"), switch("--raw")],
    ])
}
