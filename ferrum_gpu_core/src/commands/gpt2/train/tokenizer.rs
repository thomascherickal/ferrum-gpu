use super::*;

/// Load `--tokenizer` if the file exists, else train one on `text` (and save it there).
pub(super) fn tokenizer(
    a: &Args,
    text: &str,
    vocab: usize,
    out: &mut dyn Write,
) -> Result<ByteBpeTokenizer, String> {
    if let Some(p) = a.get("--tokenizer") {
        if Path::new(p).exists() {
            let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            let t = tokenizer_gguf::from_gguf_bytes(&bytes).map_err(|e| format!("{p}: {e}"))?;
            say(out, format!("tokenizer: {p} ({} tokens)", t.vocab_size()))?;
            return Ok(t);
        }
    }
    // --vocab 0: character-level, as one token per byte (256 ids, no merges). Pure ASCII text is
    // exactly character-level; other characters become their UTF-8 bytes. Unlike ferrum-cpu's own
    // character model (a distinct-character vocabulary in a file only it reads), this one is an
    // ordinary tokenizer file and an ordinary `gpt2` GGUF, so it exports and loads everywhere.
    let t = if vocab == 0 {
        ByteBpeTokenizer::byte_level()
    } else {
        ByteBpeTokenizer::train_with_docs(text, vocab).map_err(|e| format!("tokenizer: {e}"))?
    };
    let kind = if vocab == 0 {
        "character-level (bytes)"
    } else {
        "trained"
    };
    say(out, format!("tokenizer: {kind} {} tokens", t.vocab_size()))?;
    if let Some(p) = a.get("--tokenizer") {
        atomic_write(Path::new(p), &tokenizer_gguf::to_gguf_bytes(&t))
            .map_err(|e| format!("{p}: {e}"))?;
        say(out, format!("wrote {p}"))?;
    }
    Ok(t)
}

/// The tokenizer for a token file: `--tokenizer` must exist and be the file the ids were made with.
pub(super) fn token_file_tokenizer(
    a: &Args,
    tf: &TokenFile,
    out: &mut dyn Write,
) -> Result<ByteBpeTokenizer, String> {
    let p = a
        .get("--tokenizer")
        .filter(|p| Path::new(p).exists())
        .ok_or("a token file needs --tokenizer with the tokenizer file it was made with")?;
    let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
    if sha256_hex(&bytes) != tf.tokenizer_sha256() {
        return Err(format!(
            "{p} is not the tokenizer this token file was made with (sha256 {} != {})",
            sha256_hex(&bytes),
            tf.tokenizer_sha256()
        ));
    }
    let t = tokenizer_gguf::from_gguf_bytes(&bytes).map_err(|e| format!("{p}: {e}"))?;
    say(out, format!("tokenizer: {p} ({} tokens)", t.vocab_size()))?;
    Ok(t)
}

/// A short identity of the tokenizer, recorded in checkpoints so a resume cannot mix vocabularies.
pub(super) fn corpus_id(t: &ByteBpeTokenizer) -> String {
    let h = t
        .encode_state()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("bpe-{}-{h:016x}", t.vocab_size())
}
