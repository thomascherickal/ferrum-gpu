// Provenance: the gradient formulas (LayerNorm, attention, GELU derivative incl. its |x| >= 10
// saturation) follow ferrum-cpu `train_transformer/kernels.rs`, `backward.rs` and `activation.rs`
// (commit 56a9d3a); rewritten over plain slices, ours from here on.
//! Training forward/backward for [`Gpt2`] on the CPU: the oracle for the GPU trainer and the
//! CPU baseline for speed comparisons. Batched: `ids` is `batch` sequences of length `t`.
use crate::dropout::Dropout;
use crate::gpt2::{Block, Gpt2};
use crate::matmul::MatmulError;
use crate::nn::{self, LN_EPS};
use crate::tuned::matmul_tuned;
mod mm;
pub use mm::*;
mod attention_bwd;
pub use attention_bwd::*;
mod block_backward;
mod block_pass;
mod check;
use block_pass::Pass;
pub use check::*;
mod loss_and_grads_with;
pub use loss_and_grads_with::*;

#[cfg(test)]
mod fd_check;
#[cfg(test)]
pub(crate) use fd_check::check_tensor_fd;
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
// index loops read clearer in element-wise finite-difference checks
mod tests;
