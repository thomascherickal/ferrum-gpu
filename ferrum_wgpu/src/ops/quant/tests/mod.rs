use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, REDUCING};
use ferrum_gguf::quant::decode;
use ferrum_ref::matmul::{matmul, rand_vec};

const KINDS: [QKind; 3] = [QKind::Q8_0, QKind::Q4_0, QKind::F16];

/// Values that stress the quantisers: spread magnitudes, all-zero blocks, tiny blocks (the f16
/// scale is subnormal), large blocks and exact small integers.
fn nasty(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = rand_vec(n, 9).into_iter().map(|x| x * 3.0).collect();
    for (i, x) in v.iter_mut().enumerate() {
        match i / 32 % 7 {
            0 => *x *= 1e-6,
            1 => *x = 0.0,
            2 => *x *= 1e4,
            3 => *x = (i % 9) as f32 - 4.0,
            _ => {}
        }
    }
    v
}

/// What the CPU sees for a quantised `[k, n]` weight: decode of the GGUF bytes, `[n, k]` rows.
fn decoded_rows(kind: QKind, bytes: &[u8], k: usize, n: usize) -> Vec<f32> {
    decode(kind.ggml().unwrap(), bytes, k * n).unwrap()
}

mod dequantised_values_are_bit_identical_to_the_cpu_decode;
mod sizes_and_eligibility;
mod the_kernel_choice_follows_the_measurements;
mod the_packed_weight_is_smaller_by_the_claimed_ratio;
