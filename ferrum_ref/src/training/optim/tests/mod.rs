use super::*;
use crate::gpt2::Config;
use crate::grad::loss_and_grads;

fn cfg() -> Config {
    Config {
        vocab: 9,
        ctx: 6,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 12,
    }
}

mod first_step_moves_each_parameter_by_about_lr;
mod schedule_warms_up_peaks_and_decays_to_zero;
