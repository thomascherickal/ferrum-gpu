use super::*;
use crate::gpt2::{Config, Gpt2};

mod a_prefix_hides_the_held_out_tail;
mod a_tied_trainer_keeps_the_head_exactly_the_embedding_transposed_and_still_learns;
mod an_explicit_stride_sets_the_window_starts;
mod cfg;
mod checkpoints_follow_the_cadence_and_always_come_at_the_end;
mod invalid_configs_and_corpora_are_refused;
mod qat_trains_the_masters_through_the_snapped_weights;
mod resume_refuses_a_changed_run_shape;
mod resuming_from_a_snapshot_equals_an_uninterrupted_run;
mod schedule_and_budgets_are_applied;
mod the_cpu_engine_learns_a_periodic_corpus;
mod windows_start_every_stride_while_a_target_remains;
use cfg::*;
mod failing;
use failing::*;
