use super::*;
use crate::writer::GgufBuilder;
use crate::GGML_F32;

fn sample() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.meta("gpt2.block_count", Value::U32(2));
    b.meta("x.eps", Value::F32(1e-5));
    b.meta(
        "x.list",
        Value::Arr(vec![Value::Str("a".into()), Value::Str("bc".into())]),
    );
    b.tensor_f32("w", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    b.tensor_f32("b", &[2], &[0.5, -0.5]);
    b.into_bytes()
}

mod a_header_only_parse_checks_extents_against_the_file_length;
mod hostile_array_and_tensor_sizes_are_rejected;
