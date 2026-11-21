use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, REDUCING};
use ferrum_ref::grad;
use ferrum_ref::matmul::{matmul, rand_vec};
use ferrum_ref::nn;

fn show(name: &str, shape: &str, e: f32, tol: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
    assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
}

fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

mod attention_bwd_matches_cpu;
mod backward_kernels_are_bit_identical_across_runs;
mod embedding_grad_with_duplicate_tokens_matches_cpu;
mod layernorm_bwd_matches_cpu;
mod linear_bwd_matches_cpu;
mod matmul_ex_matches_cpu_for_all_four_transpose_combinations;
mod multi_command_backward_ops_report_a_fault_at_each_of_their_commands;
