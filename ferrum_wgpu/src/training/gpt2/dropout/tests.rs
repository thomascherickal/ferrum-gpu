//! Dropout on the GPU: the kernel's masks equal the CPU's bit for bit, and a GPU run with dropout
//! follows the CPU run. Each skips, printing why, when no adapter opens.

mod mask;
mod parity;
mod state;

use ferrum_ref::gpt2::Config;

/// A small model; `hidden` is deliberately not a multiple of the workgroup size.
pub(super) fn cfg() -> Config {
    Config {
        vocab: 24,
        ctx: 8,
        dim: 16,
        heads: 2,
        layers: 2,
        hidden: 40,
    }
}

/// `batch` sequences of `t` tokens and their targets.
pub(super) fn data(c: &Config, batch: usize, t: usize, salt: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1 + salt) % c.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2 + salt) % c.vocab).collect(),
    )
}
