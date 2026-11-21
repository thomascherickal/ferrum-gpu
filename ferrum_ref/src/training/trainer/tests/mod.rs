use super::*;
use crate::gpt2::{Config, Gpt2};

mod a_prefix_hides_the_held_out_tail;
mod a_tied_trainer_keeps_the_head_exactly_the_embedding_transposed_and_still_learns;
mod cfg;
mod checkpoints_follow_the_cadence_and_always_come_at_the_end;
mod qat_trains_the_masters_through_the_snapped_weights;
mod resume_refuses_a_changed_run_shape;
mod windows_start_every_stride_while_a_target_remains;
use cfg::*;
mod failing;
use failing::*;
