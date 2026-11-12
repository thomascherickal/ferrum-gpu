//! Tied heads on the GPU: the transpose kernel, parity with the CPU tied trainer, and the state
//! rules (when the head is re-derived). Each skips, printing why, when no adapter opens.

mod kernel;
mod parity;
mod state;

use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::grad::transpose;

/// A small model whose embedding and head are wide enough to cross a workgroup boundary.
pub(super) fn cfg() -> Config {
    Config {
        vocab: 24,
        ctx: 8,
        dim: 16,
        heads: 2,
        layers: 1,
        hidden: 32,
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

/// Is `w`'s head exactly its embedding transposed?
pub(super) fn is_tied(w: &Gpt2) -> bool {
    w.head == transpose(&w.tok, w.cfg.vocab, w.cfg.dim)
}
