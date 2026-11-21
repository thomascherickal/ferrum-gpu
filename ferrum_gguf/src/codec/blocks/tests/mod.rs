// The hand-computed cases below follow the unit tests in ferrum-cpu's dequant.rs (commit
// dbf6392c8883f293ac64b28a29c55e823179b257): constant blocks whose answer follows from the format.
use super::*;
use crate::f16::f32_to_f16;

fn f(v: f32) -> [u8; 2] {
    f32_to_f16(v).to_le_bytes()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

/// 12 packed bytes that unpack to scale 2, min 0 for all eight sub-blocks.
const SCALES_D2_M0: [u8; 12] = [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2];

mod layout_covers_exactly_the_read_only_types;
mod q6_k_scales_are_signed_and_sub_blocks_are_independent;
mod scale_min_unpacks_every_sub_block;
