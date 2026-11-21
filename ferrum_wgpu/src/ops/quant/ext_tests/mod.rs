//! The weight formats added after Q8_0, Q4_0 and F16: MXFP4, FP8 (E4M3, E5M2) and Q2_K.
use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, REDUCING};
use ferrum_gguf::f16::f32_to_f16;
use ferrum_ref::matmul::{matmul, rand_vec};

pub(super) const EXT: [QKind; 4] = [QKind::Fp4, QKind::Fp8E4M3, QKind::Fp8E5M2, QKind::Q2K];

/// Shapes `(k, n)` that fill whole blocks of `kind`, including awkward widths for the kernels.
fn shapes(kind: QKind) -> Vec<(usize, usize)> {
    if kind == QKind::Q2K {
        vec![(256, 1), (256, 5), (512, 17)]
    } else {
        vec![(32, 1), (32, 5), (64, 3), (96, 65)]
    }
}

fn weights(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = rand_vec(n, 21).into_iter().map(|x| x * 2.0).collect();
    for (i, x) in v.iter_mut().enumerate() {
        match i / 32 % 5 {
            0 => *x *= 1e-5,
            1 => *x = 0.0,
            2 => *x *= 300.0,
            _ => {}
        }
    }
    v
}

fn rows_of(kind: QKind, bytes: &[u8], k: usize, n: usize) -> Vec<f32> {
    dequantise_bytes(kind, bytes, k * n).unwrap()
}

mod linear_matches_the_cpu_matmul_on_the_dequantised_weights;
mod sizes_eligibility_and_the_memory_ratio;
