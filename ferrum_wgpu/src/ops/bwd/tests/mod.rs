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

/// `(dx, dgain, dshift)` of the device LayerNorm backward over `rows x cols`.
fn gpu_layernorm_bwd(
    g: &Gpu,
    (x, gain, dy): (&[f32], &[f32], &[f32]),
    (rows, cols): (usize, usize),
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (gx, gg, gdy) = (
        g.upload(x).unwrap(),
        g.upload(gain).unwrap(),
        g.upload(dy).unwrap(),
    );
    let (gdx, gdg, gdb, st) = (
        g.zeros(rows * cols).unwrap(),
        g.zeros(cols).unwrap(),
        g.zeros(cols).unwrap(),
        g.zeros(2 * rows).unwrap(),
    );
    layernorm_bwd(g, (&gx, &gg, &gdy), (&gdx, &gdg, &gdb), &st, (rows, cols)).unwrap();
    (
        g.read(&gdx).unwrap(),
        g.read(&gdg).unwrap(),
        g.read(&gdb).unwrap(),
    )
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
