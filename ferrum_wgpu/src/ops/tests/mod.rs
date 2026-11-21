use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, EXACT, REDUCING};
use ferrum_ref::matmul::rand_vec;
use ferrum_ref::nn;

fn show(name: &str, shape: &str, e: f32, tol: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
    assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
}

/// Scale random data so GELU/softmax see a useful range, including large magnitudes.
fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

mod add_and_bias_match_cpu;
mod attention_decode_matches_the_last_row_of_full_attention;
mod attention_matches_cpu;
mod cross_entropy_matches_cpu;
mod gelu_matches_cpu_including_saturation;
mod multi_command_ops_report_a_fault_at_each_of_their_commands;
mod shape_mismatches_are_errors_not_dispatches;
