use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, EXACT, REDUCING};
use ferrum_ref::llama as cpu;
use ferrum_ref::matmul::rand_vec;

fn show(name: &str, shape: &str, e: f32, tol: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
    assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
}

fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

/// CPU reference for full GQA attention: per batch, per row, `attend_row` over that row's prefix.
fn cpu_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (batch, t, heads, kv, hd): (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let (qd, kvd) = (heads * hd, kv * hd);
    let mut out = Vec::new();
    for b in 0..batch {
        for i in 0..t {
            let r = b * t + i;
            out.extend(cpu::attend_row(
                &q[r * qd..(r + 1) * qd],
                &k[b * t * kvd..],
                &v[b * t * kvd..],
                i + 1,
                (heads, kv, hd),
            ));
        }
    }
    out
}

mod attention_refuses_bad_shapes;
mod gqa_attention_matches_cpu_for_every_group_ratio;
mod rmsnorm_matches_cpu_on_odd_shapes;
mod rope_refuses_bad_shapes;
