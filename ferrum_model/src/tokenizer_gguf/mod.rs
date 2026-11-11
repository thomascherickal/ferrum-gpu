//! Tokenizer files and model metadata: the GGUF side of [`ferrum_tok::ByteBpeTokenizer`].
//!
//! Two layouts, both the ones ferrum-cpu writes (so a tokenizer file or a model from either side
//! loads on the other):
//! - a standalone **tokenizer file**: `general.architecture = "ferrum-tokenizer"`, the three
//!   `tokenizer.ggml.{tokens,merges,token_type}` arrays (a valid llama.cpp vocabulary) and
//!   `tokenizer.ferrum.state`, the lossless source of truth;
//! - the same keys inside a **model file**, read by [`from_gguf`] when the model carries them.

use ferrum_gguf::{Gguf, GgufBuilder, Value};
use ferrum_tok::ByteBpeTokenizer;

/// Architecture string of a standalone tokenizer file.
pub const TOKENIZER_ARCH: &str = "ferrum-tokenizer";
/// Metadata key holding [`ByteBpeTokenizer::encode_state`].
pub const STATE_KEY: &str = "tokenizer.ferrum.state";

fn strings(v: Vec<String>) -> Value {
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

fn str_array(g: &Gguf, key: &str) -> Result<Vec<String>, String> {
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

fn i32_array(g: &Gguf, key: &str) -> Result<Vec<i32>, String> {
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

/// Read the tokenizer out of a parsed GGUF (a tokenizer file or a model carrying the keys).
/// The state string, when present, is authoritative and cross-checked against the arrays; a
/// file without it is rebuilt from the arrays alone.
pub fn from_gguf(g: &Gguf) -> Result<ByteBpeTokenizer, String> {
    let tokens = str_array(g, "tokenizer.ggml.tokens")?;
    let merges = str_array(g, "tokenizer.ggml.merges")?;
    let types = i32_array(g, "tokenizer.ggml.token_type")?;
    let Some(Value::Str(state)) = g.get(STATE_KEY) else {
        return ByteBpeTokenizer::from_gguf_vocab(&tokens, &merges, &types)
            .map_err(|e| e.to_string());
    };
    let tok = ByteBpeTokenizer::from_state(state).map_err(|e| e.to_string())?;
    if tok.gguf_tokens() != tokens || tok.gguf_merges() != merges || tok.gguf_token_types() != types
    {
        return Err(format!(
            "tokenizer data is corrupt: {STATE_KEY} and the tokenizer.ggml.* arrays disagree"
        ));
    }
    Ok(tok)
}

/// Read a standalone tokenizer file.
pub fn from_gguf_bytes(bytes: &[u8]) -> Result<ByteBpeTokenizer, String> {
    if bytes.len() < 4 {
        return Err("empty or truncated tokenizer file".into());
    }
    if !bytes.starts_with(b"GGUF") {
        return Err("not a GGUF tokenizer file".into());
    }
    let g = Gguf::parse(bytes.to_vec()).map_err(|e| e.to_string())?;
    match g.string("general.architecture") {
        Ok(TOKENIZER_ARCH) => from_gguf(&g),
        other => Err(format!(
            "not a {TOKENIZER_ARCH} file (general.architecture = {other:?})"
        )),
    }
}

#[cfg(test)]
mod tests;
