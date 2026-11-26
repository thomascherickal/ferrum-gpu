use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_ref::gpt2::Gpt2;

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 12,
        heads: 3,
        layers: 2,
        hidden: 20,
    }
}

mod a_fault_at_any_command_fails_decoding_cleanly_and_the_context_does_not_advance;
mod a_gguf_exported_model_runs_identically_after_a_round_trip;
mod an_invalid_config_or_a_failed_allocation_is_refused_at_construction;
mod reset_restarts_the_context_and_gives_the_same_answers;
