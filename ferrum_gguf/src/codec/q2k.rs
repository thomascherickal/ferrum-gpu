//! Q2_K (ggml type 10) encoder: 256-value super-blocks of sixteen 16-value sub-blocks, 2-bit
//! quants with a 4-bit scale and a 4-bit min per sub-block, and an f16 super scale and super min.
//! 84 bytes per block: `scales[16]`, `qs[64]`, `d`, `dmin` (decoded by `blocks::dequant`).
//!
//! The search for each sub-block's scale and min follows ggml's reference quantiser
//! (`quantize_row_q2_K_ref` with `make_qkx2_quants`, no importance matrix): the same candidate
//! scales, the same least-squares fit, the same rounding, in f32.
use crate::f16::{f16_to_f32, f32_to_f16};
use crate::QK_K;
mod block;
pub use block::*;
mod encode;
pub use encode::*;

#[cfg(test)]
mod tests;
