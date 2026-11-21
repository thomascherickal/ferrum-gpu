use super::*;

/// A parsed imported model with its tokenizer and (if it is an instruct model) chat format.
pub struct Loaded {
    /// The weights.
    pub model: Llama,
    /// The file's own tokenizer.
    pub tok: GgufTokenizer,
    /// The chat format, when the file is recognised as an instruct model.
    pub chat: Option<ChatFormat>,
}

pub(super) fn parse(path: &str) -> Result<Gguf, String> {
    Gguf::parse(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))
}

/// Does the file carry a vocabulary? (`false` for unreadable files; the real error comes later.)
// ponytail: reads the whole file; a header-only probe (`Gguf::parse_prefix`) would avoid a second
// read of a large model.
pub fn has_tokenizer(path: &str) -> bool {
    parse(path).is_ok_and(|g| g.get("tokenizer.ggml.tokens").is_some())
}

/// Load the weights, the tokenizer and the chat format.
pub fn load(path: &str) -> Result<Loaded, String> {
    let g = parse(path)?;
    let model = from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    let tok = GgufTokenizer::from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    if tok.vocab_size() > model.cfg.vocab {
        return Err(format!(
            "{path}: the tokenizer has {} tokens but the model only {}",
            tok.vocab_size(),
            model.cfg.vocab
        ));
    }
    let arch = g.string("general.architecture").ok();
    let template = g.string("tokenizer.chat_template").ok();
    let chat = detect(template, arch, |p| tok.token_id(p).is_some());
    Ok(Loaded { model, tok, chat })
}

/// The prompt ids and whether they were wrapped in the chat template.
pub(super) fn prompt_ids(l: &Loaded, a: &Args) -> Result<(Vec<usize>, bool), String> {
    if let Some(s) = a.get("--ids") {
        let ids = s
            .split_whitespace()
            .map(|x| {
                x.parse::<usize>()
                    .map_err(|_| format!("--ids: '{x}' is not an id"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((ids, false));
    }
    let text = a.need("--prompt")?;
    let mut ids: Vec<usize> = l.tok.prefix_bos().into_iter().collect();
    match l.chat.filter(|_| !a.has("--raw")) {
        Some(f) => {
            ids.extend(encode_segments(&l.tok, &render(f, text)));
            Ok((ids, true))
        }
        None => {
            ids.extend(l.tok.encode(text));
            Ok((ids, false))
        }
    }
}

/// The text of `new`, decoded after the token `anchor` so a leading space survives
/// SentencePiece's dummy-prefix stripping.
pub(super) fn continuation(tok: &GgufTokenizer, anchor: usize, new: &[usize]) -> String {
    let mut ids = vec![anchor];
    ids.extend_from_slice(new);
    let full = tok.decode(&ids);
    let base = tok.decode(&[anchor]);
    full.strip_prefix(&base)
        .map_or(full.clone(), str::to_string)
}
