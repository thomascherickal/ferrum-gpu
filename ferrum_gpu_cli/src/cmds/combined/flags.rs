//! The flag tables of `run`'s two halves.

use crate::cmds::{gpt2, table, DEVICE, GEN, TRAIN};
use crate::flags::{switch, val, Flag};

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
