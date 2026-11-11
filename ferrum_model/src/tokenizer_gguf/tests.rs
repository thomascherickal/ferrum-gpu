use super::*;
use ferrum_tok::ByteBpeTokenizer;

// Golden files made by ferrum-cpu (commit 56a9d3a, see fixtures/README.md).
const TOK: &[u8] = include_bytes!("../../../fixtures/bpe/tok.gguf");
const MODEL: &[u8] = include_bytes!("../../../fixtures/bpe/model.gguf");
const CORPUS: &str = include_str!("../../../fixtures/bpe/corpus.txt");

fn small() -> ByteBpeTokenizer {
    ByteBpeTokenizer::train(&"abab cdcd ünï low lower ".repeat(10), 300).unwrap()
}

#[test]
fn round_trips_through_a_tokenizer_file() {
    let t = small();
    let back = from_gguf_bytes(&to_gguf_bytes(&t)).unwrap();
    assert_eq!(back.encode_state(), t.encode_state());
    assert_eq!(back.encode("abab ünï"), t.encode("abab ünï"));
}

#[test]
fn reads_a_ferrum_cpu_tokenizer_file() {
    let t = from_gguf_bytes(TOK).unwrap();
    assert_eq!(t.vocab_size(), 320);
    let text = "Der Hund läuft 日本語 🌸 it's";
    assert_eq!(t.decode(&t.encode(text)), text);
    // writing it back gives a file that reads back to the same tokenizer
    assert_eq!(
        from_gguf_bytes(&to_gguf_bytes(&t)).unwrap().encode_state(),
        t.encode_state()
    );
}

#[test]
fn a_model_file_from_ferrum_cpu_carries_the_same_tokenizer() {
    let g = Gguf::parse(MODEL.to_vec()).unwrap();
    let from_model = from_gguf(&g).unwrap();
    let from_file = from_gguf_bytes(TOK).unwrap();
    assert_eq!(from_model.encode_state(), from_file.encode_state());
}

#[test]
fn rebuilds_from_arrays_when_the_state_key_is_absent() {
    let t = small();
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKENIZER_ARCH.into()));
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta("tokenizer.ggml.tokens", strings(t.gguf_tokens()));
    b.meta("tokenizer.ggml.merges", strings(t.gguf_merges()));
    b.meta(
        "tokenizer.ggml.token_type",
        Value::Arr(t.gguf_token_types().into_iter().map(Value::I32).collect()),
    );
    let back = from_gguf_bytes(&b.into_bytes()).unwrap();
    assert_eq!(back.encode_state(), t.encode_state());
}

#[test]
fn rejects_bad_files() {
    assert!(from_gguf_bytes(b"").unwrap_err().contains("truncated"));
    assert!(from_gguf_bytes(b"not gguf at all")
        .unwrap_err()
        .contains("not a GGUF"));
    assert!(from_gguf_bytes(&MODEL[..40]).is_err());
    // a model file is not a tokenizer file
    assert!(from_gguf_bytes(MODEL)
        .unwrap_err()
        .contains("not a ferrum-tokenizer"));
}

#[test]
fn rejects_missing_wrong_and_disagreeing_keys() {
    let t = small();
    let build = |tokens: Option<Value>, types: Option<Value>, state: Option<&str>| {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str(TOKENIZER_ARCH.into()));
        if let Some(v) = tokens {
            b.meta("tokenizer.ggml.tokens", v);
        }
        b.meta("tokenizer.ggml.merges", strings(t.gguf_merges()));
        if let Some(v) = types {
            b.meta("tokenizer.ggml.token_type", v);
        }
        if let Some(s) = state {
            b.meta(STATE_KEY, Value::Str(s.into()));
        }
        b.into_bytes()
    };
    let toks = || Some(strings(t.gguf_tokens()));
    let types = || {
        Some(Value::Arr(
            t.gguf_token_types().into_iter().map(Value::I32).collect(),
        ))
    };
    assert!(from_gguf_bytes(&build(None, types(), None))
        .unwrap_err()
        .contains("tokens"));
    assert!(from_gguf_bytes(&build(toks(), None, None))
        .unwrap_err()
        .contains("token_type"));
    let not_i32 = Some(Value::Arr(vec![Value::Str("x".into())]));
    assert!(from_gguf_bytes(&build(toks(), not_i32, None))
        .unwrap_err()
        .contains("i32"));
    // state disagrees with the arrays
    let other = ByteBpeTokenizer::byte_level().encode_state();
    let e = from_gguf_bytes(&build(toks(), types(), Some(&other))).unwrap_err();
    assert!(e.contains("corrupt"), "{e}");
    // state that does not parse
    assert!(from_gguf_bytes(&build(toks(), types(), Some("x,y"))).is_err());
}

#[test]
fn encodes_the_corpus_to_the_ids_ferrum_cpu_wrote() {
    // ferrum-cpu `tokenize corpus.txt --tokenizer tok.gguf`: 942 ids, 31 documents, I16 ids.
    let g = Gguf::parse(include_bytes!("../../../fixtures/bpe/corpus.tok.gguf").to_vec()).unwrap();
    assert_eq!(g.string("general.architecture").unwrap(), "ferrum-tokens");
    assert_eq!(g.uint("ferrum.tokens.token_count").unwrap(), 942);
    let want: Vec<usize> = g
        .tensor_bytes("tokens")
        .unwrap()
        .chunks_exact(2)
        .map(|c| usize::from(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let tok = from_gguf_bytes(TOK).unwrap();
    assert_eq!(tok.encode_docs(CORPUS), want);
    assert_eq!(
        g.uint("ferrum.tokens.eos_id").unwrap(),
        tok.eos_id().unwrap()
    );
}

#[test]
fn our_gpt2_reader_loads_the_ferrum_cpu_model_file() {
    let g = Gguf::parse(MODEL.to_vec()).unwrap();
    let m = crate::gpt2_gguf::from_gguf(&g).unwrap();
    // vocab 320 BPE tokens, context 64 (printed by ferrum-cpu when it wrote the file)
    assert_eq!(m.cfg.vocab, 320);
    assert_eq!(m.cfg.ctx, 64);
}

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
