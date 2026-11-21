use super::*;

pub(crate) fn tiny() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 12,
    }
}

mod cached_feed_matches_the_full_forward_at_every_position;
mod decay_flags_match_the_matrix_rule;
mod forward_rejects_bad_input;
