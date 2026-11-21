use super::*;

/// A parsed model file: weights and tokenizer.
pub struct Loaded {
    /// The model.
    pub model: Gpt2,
    /// Its tokenizer (byte-level for a vocabulary of 256 without stored tokens).
    pub tok: ByteBpeTokenizer,
}

pub(super) fn parse(path: &str) -> Result<Gguf, String> {
    Gguf::parse(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))
}

/// Does `path` hold a `gpt2` model? (`false` for unreadable files; the real error comes later.)
pub fn is_gpt2(path: &str) -> bool {
    parse(path).is_ok_and(|g| g.string("general.architecture") == Ok("gpt2"))
}

/// Load a GPT-2 file and its tokenizer.
pub fn load(path: &str) -> Result<Loaded, String> {
    let g = parse(path)?;
    let model = from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    let tok = if g.get("tokenizer.ggml.tokens").is_some() {
        tokenizer_gguf::from_gguf(&g).map_err(|e| format!("{path}: {e}"))?
    } else if model.cfg.vocab == 256 {
        ByteBpeTokenizer::byte_level()
    } else {
        return Err(format!(
            "{path}: the file has no tokenizer for its {} tokens",
            model.cfg.vocab
        ));
    };
    if tok.vocab_size() != model.cfg.vocab {
        return Err(format!(
            "{path}: the tokenizer has {} tokens but the model {}",
            tok.vocab_size(),
            model.cfg.vocab
        ));
    }
    Ok(Loaded { model, tok })
}

/// The `model:` line.
pub fn shape_of(path: &str) -> Result<String, String> {
    let c = load(path)?.model.cfg;
    Ok(format!(
        "model: gpt2 vocab {} ctx {} dim {} heads {} layers {} hidden {}",
        c.vocab, c.ctx, c.dim, c.heads, c.layers, c.hidden
    ))
}

/// Sampling and stopping options of one generation.
pub struct Opts {
    /// Most tokens to add.
    pub max_new: usize,
    /// Temperature, top-k, top-p, repetition penalty.
    pub params: Params,
    /// Sampler seed.
    pub seed: u64,
    /// Stop strings: the text ends before the first one found.
    pub stops: Vec<String>,
    /// Emit text as it is produced.
    pub stream: bool,
    /// Token ids that end the turn (end-of-sequence, `<|im_end|>`): sampling one stops, unprinted.
    pub stop_ids: Vec<usize>,
}

/// Parse the sampling and stopping flags of a generation.
pub fn opts(a: &Args) -> Result<Opts, String> {
    let name = |names: &[&'static str], d: &'static str| a.first(names).unwrap_or(d);
    let max_new = a
        .first(&["--max-new", "--max-tokens", "--chars"])
        .map_or(Ok(64usize), |n| a.num(n, 64))?;
    Ok(Opts {
        max_new,
        params: Params {
            temperature: a.num(name(&["--temperature", "--temp"], "--temperature"), 0.0f32)?,
            top_k: a.num("--top-k", 0)?,
            top_p: a.num("--top-p", 1.0f32)?,
            repetition_penalty: a.num("--repetition-penalty", 1.0f32)?,
        },
        seed: a.num(name(&["--seed", "--gen-seed"], "--seed"), 1u64)?,
        stops: a.all("--stop").to_vec(),
        stream: a.has("--stream"),
        stop_ids: vec![],
    })
}
