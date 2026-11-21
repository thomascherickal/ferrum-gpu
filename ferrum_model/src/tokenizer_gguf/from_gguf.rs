use super::*;

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
