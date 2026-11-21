//! When the head is re-derived, and what a tied head leaves untouched.
use super::{cfg, data, is_tied};
use crate::device::open_for_test;
use crate::train::{Engine, GpuTrainer, StepConfig};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Adam;

fn sc() -> StepConfig {
    StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    }
}

mod switching_tying_on_ties_the_head_at_once;
mod untying_stops_re_deriving_the_head;
