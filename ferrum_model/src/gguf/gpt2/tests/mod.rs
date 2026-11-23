use super::*;
use ferrum_gguf::{GGML_F32, GGML_Q8_0};

fn cfg(dim: usize) -> Config {
    Config {
        vocab: 13,
        ctx: 8,
        dim,
        heads: 2,
        layers: 2,
        hidden: dim * 2,
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

mod a_file_without_the_embedding_is_refused;
mod an_embedding_whose_width_disagrees_with_the_metadata_is_refused;
mod quantised_models_load_and_stay_close;
mod transpose_is_correct_and_an_involution;
