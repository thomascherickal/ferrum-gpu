//! G2: a model whose tensors are larger than the (lowered) binding limit is stored as blocks and
//! must give the same answers as the same weights stored whole.
use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_ref::llama::RopeType;

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

/// Larger than the biggest buffer that cannot be split (the KV cache is 640 bytes), smaller than
/// every weight: the embedding (2176 bytes) and each matrix (4096 to 8192) are cut into blocks.
const LIMIT: u64 = 1000;
const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5];

fn blocks_of(m: &GpuLlama) -> (usize, usize, usize) {
    let head = match &m.head {
        Lin::Sharded { parts, .. } => parts.len(),
        _ => 1,
    };
    let gate = match &m.blocks[0].gate {
        Lin::Sharded { parts, .. } => parts.len(),
        _ => 1,
    };
    (m.tok.parts(), head, gate)
}

mod an_allocation_failure_while_uploading_blocks_is_reported;
mod sharded_f32_decode_equals_the_whole_model_with_and_without_biases;
mod what_cannot_be_split_still_fails_the_binding_check_and_says_which;
