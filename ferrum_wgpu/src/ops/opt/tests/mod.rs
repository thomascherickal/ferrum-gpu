use super::*;
use crate::device::open_for_test;
use crate::tolerance::max_rel_err;
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::matmul::rand_vec;
use ferrum_ref::optim::{adam_step, clip_scale, global_norm, AdamState};

fn acc_src(g: &Gpu, n: usize) -> GpuBuf {
    g.zeros(n).unwrap()
}

mod adam_update_matches_the_cpu_over_several_steps_with_and_without_decay;
mod the_dispatch_limit_is_enforced_without_needing_a_huge_tensor;
