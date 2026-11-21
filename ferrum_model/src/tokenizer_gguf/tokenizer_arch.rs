use super::*;

/// Architecture string of a standalone tokenizer file.
pub const TOKENIZER_ARCH: &str = "ferrum-tokenizer";
/// Metadata key holding [`ByteBpeTokenizer::encode_state`].
pub const STATE_KEY: &str = "tokenizer.ferrum.state";

pub(super) fn strings(v: Vec<String>) -> Value {
    Value::Arr(v.into_iter().map(Value::Str).collect())
}

/// Add the `tokenizer.*` keys to a builder (for a model file or a tokenizer file).
pub fn put_metadata(b: &mut GgufBuilder, tok: &ByteBpeTokenizer) {
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta("tokenizer.ggml.tokens", strings(tok.gguf_tokens()));
    b.meta("tokenizer.ggml.merges", strings(tok.gguf_merges()));
    b.meta(
        "tokenizer.ggml.token_type",
        Value::Arr(tok.gguf_token_types().into_iter().map(Value::I32).collect()),
    );
    b.meta(STATE_KEY, Value::Str(tok.encode_state()));
}

/// The keys a model file carries for its tokenizer: the llama.cpp vocabulary arrays, the
/// pre-tokenizer name and the special-token ids (`bos` = `eos` = the end-of-text token, as
/// ferrum-cpu writes them). Unlike a standalone tokenizer file there is no `tokenizer.ferrum.state`.
pub fn put_vocab(b: &mut GgufBuilder, tok: &ByteBpeTokenizer) {
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta("tokenizer.ggml.pre", Value::Str("gpt-2".into()));
    b.meta("tokenizer.ggml.tokens", strings(tok.gguf_tokens()));
    b.meta("tokenizer.ggml.merges", strings(tok.gguf_merges()));
    b.meta(
        "tokenizer.ggml.token_type",
        Value::Arr(tok.gguf_token_types().into_iter().map(Value::I32).collect()),
    );
    if let Some(eos) = tok.eos_id() {
        b.meta("tokenizer.ggml.bos_token_id", Value::U32(eos as u32));
        b.meta("tokenizer.ggml.eos_token_id", Value::U32(eos as u32));
    }
    b.meta("tokenizer.ggml.add_bos_token", Value::Bool(false));
    b.meta("tokenizer.ggml.add_eos_token", Value::Bool(false));
}

/// A standalone tokenizer file.
pub fn to_gguf_bytes(tok: &ByteBpeTokenizer) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKENIZER_ARCH.into()));
    put_metadata(&mut b, tok);
    b.into_bytes()
}

pub(super) fn str_array(g: &Gguf, key: &str) -> Result<Vec<String>, String> {
    match g.get(key) {
        Some(Value::Arr(a)) => Ok(a
            .iter()
            .filter_map(|v| match v {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect()),
        _ => Err(format!("key '{key}' is missing or not an array")),
    }
}

pub(super) fn i32_array(g: &Gguf, key: &str) -> Result<Vec<i32>, String> {
    match g.get(key) {
        Some(Value::Arr(a)) => a
            .iter()
            .map(|v| match v {
                Value::I32(x) => Ok(*x),
                _ => Err(format!("key '{key}' is not an i32 array")),
            })
            .collect(),
        _ => Err(format!("key '{key}' is missing or not an array")),
    }
}
