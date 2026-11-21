use super::*;
use crate::device::open_for_test;
use crate::ops_llama::rope;
use crate::tolerance::{max_rel_err, BACKWARD};
use ferrum_ref::llama as cpu;
use ferrum_ref::llama::{RopeScaling, RopeType};
use ferrum_ref::matmul::rand_vec;

fn show(name: &str, shape: &str, e: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {BACKWARD:.0e})");
    assert!(
        e < BACKWARD,
        "{name} {shape}: max rel err {e} >= {BACKWARD}"
    );
}

fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

mod bad_shapes_are_refused_and_multi_command_ops_fail_cleanly;
mod gqa_attention_bwd_matches_cpu_for_every_group_ratio;
mod rmsnorm_bwd_matches_cpu_on_odd_shapes;
mod rope_with_a_negated_sine_table_is_the_cpu_backward;
