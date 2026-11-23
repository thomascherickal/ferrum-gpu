use super::*;

fn data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
        .collect()
}

mod absurd_element_counts_overflow_the_byte_size;
mod q8_0_error_is_within_half_a_step_of_the_block_scale;
mod read_only_block_types_have_sizes_and_decode_but_cannot_be_encoded;
