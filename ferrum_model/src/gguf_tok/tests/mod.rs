use super::*;
use ferrum_gguf::GgufBuilder;

fn arr(items: &[&str]) -> Value {
    Value::Arr(items.iter().map(|s| Value::Str((*s).into())).collect())
}

fn build(
    model: &str,
    tokens: &[&str],
    merges: &[&str],
    extra: &[(&str, Value)],
) -> Result<GgufTokenizer, String> {
    let mut b = GgufBuilder::new();
    b.meta("tokenizer.ggml.model", Value::Str(model.into()));
    b.meta("tokenizer.ggml.tokens", arr(tokens));
    if !merges.is_empty() {
        b.meta("tokenizer.ggml.merges", arr(merges));
    }
    for (k, v) in extra {
        b.meta(k, v.clone());
    }
    GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap())
}

fn bpe() -> GgufTokenizer {
    // ids: 0 h, 1 e, 2 l, 3 o, 4 Ġ (the space byte), 5 he, 6 ll, 7 hell, 8 hello, 9 "<|im_end|>"
    build(
        "gpt2",
        &[
            "h",
            "e",
            "l",
            "o",
            "\u{120}",
            "he",
            "ll",
            "hell",
            "hello",
            "<|im_end|>",
        ],
        &["h e", "l l", "he ll", "hell o"],
        &[
            ("tokenizer.ggml.bos_token_id", Value::U32(9)),
            ("tokenizer.ggml.eos_token_id", Value::U32(9)),
        ],
    )
    .unwrap()
}

mod add_bos_follows_the_file_then_llama_cpps_defaults;
mod bpe_applies_merges_in_rank_order_and_decodes_exactly;
mod edge_branches_leftovers_non_byte_chars_odd_metadata_and_pre_defaults;
