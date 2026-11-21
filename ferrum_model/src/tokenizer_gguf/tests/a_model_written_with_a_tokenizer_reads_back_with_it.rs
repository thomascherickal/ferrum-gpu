use super::*;

#[test]
fn a_model_written_with_a_tokenizer_reads_back_with_it() {
    use crate::gpt2_gguf::{to_gguf_with, Quant};
    use ferrum_ref::gpt2::{Config, Gpt2};
    let tok = ByteBpeTokenizer::train(&"abab cdcd ünï low lower ".repeat(10), 300).unwrap();
    let cfg = Config {
        vocab: tok.vocab_size(),
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    };
    let m = Gpt2::random(cfg, 3);
    let g = Gguf::parse(to_gguf_with(&m, Quant::F32, Some(&tok))).unwrap();
    assert_eq!(from_gguf(&g).unwrap().encode_state(), tok.encode_state());
    assert_eq!(crate::gpt2_gguf::from_gguf(&g).unwrap(), m);
    // the keys ferrum-cpu and llama.cpp look for
    for k in [
        "ferrum.format",
        "ferrum.quant",
        "ferrum.metadata",
        "ferrum.normalizer",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.bos_token_id",
    ] {
        assert!(g.get(k).is_some(), "missing {k}");
    }
    assert_eq!(g.string("ferrum.quant").unwrap(), "f32");
    // without a tokenizer the file has no vocabulary
    let plain = Gguf::parse(crate::gpt2_gguf::to_gguf(&m, Quant::F32)).unwrap();
    assert!(plain.get("tokenizer.ggml.tokens").is_none());
    assert_eq!(plain.string("ferrum.quant").unwrap(), "f32");
}

#[test]
fn our_trainer_learns_the_merges_ferrum_cpu_learned() {
    // ferrum-cpu trained tok.gguf on this corpus with --vocab 320 (docs split on <|endoftext|>).
    let ours = ByteBpeTokenizer::train_with_docs(CORPUS, 320).unwrap();
    let theirs = from_gguf_bytes(TOK).unwrap();
    assert_eq!(ours.encode_state(), theirs.encode_state());
}

#[test]
fn an_array_of_numbers_where_strings_belong_reads_as_empty() {
    use ferrum_gguf::{Gguf, GgufBuilder, Value};
    let mut b = GgufBuilder::new();
    b.meta("k", Value::Arr(vec![Value::U32(7), Value::U32(8)]));
    b.meta(
        "s",
        Value::Arr(vec![Value::Str("a".into()), Value::Str("b".into())]),
    );
    b.meta("n", Value::U32(1));
    let g = Gguf::parse(b.into_bytes()).unwrap();
    assert!(str_array(&g, "k").unwrap().is_empty());
    assert_eq!(
        str_array(&g, "s").unwrap(),
        vec!["a".to_string(), "b".to_string()]
    );
    assert!(str_array(&g, "n")
        .unwrap_err()
        .contains("missing or not an array"));
    assert!(str_array(&g, "absent").is_err());
}
