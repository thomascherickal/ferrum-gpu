use super::*;
use crate::device::open_for_test;
use ferrum_ref::matmul::rand_vec;
use ferrum_ref::qat::{snap_cols, snap_rows};

/// Values that stress the quantiser: spread magnitudes, ties, zeros, a block of denormal-scale
/// values (the f16 scale is subnormal there), and large values.
fn nasty(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = rand_vec(n, 9).into_iter().map(|x| x * 3.0).collect();
    for (i, x) in v.iter_mut().enumerate() {
        match i / 32 % 7 {
            0 => *x *= 1e-6,                // tiny: the f16 scale becomes subnormal
            1 => *x = 0.0,                  // all-zero block
            2 => *x *= 1e4,                 // large
            3 => *x = (i % 9) as f32 - 4.0, // exact small integers: many ties at .5 steps
            _ => {}
        }
    }
    v
}

fn mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

mod the_snap_is_idempotent_on_the_device;
mod values_that_round_to_zero_are_positive_zero_like_the_cpu;
