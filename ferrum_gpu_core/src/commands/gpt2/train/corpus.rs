use super::*;

/// The `--data` corpus: a token file or UTF-8 text, never both.
pub(super) struct Corpus {
    pub(super) token_file: Option<TokenFile>,
    pub(super) text: Option<String>,
}

impl Corpus {
    pub(super) fn load(a: &Args) -> Result<Corpus, String> {
        let path = a.need("--data")?;
        if is_token_file(path) {
            let token_file = Some(TokenFile::open(path)?);
            return Ok(Corpus {
                token_file,
                text: None,
            });
        }
        let bytes = std::fs::read(path).map_err(|e| format!("--data: {e}"))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| "--data: the corpus is not valid UTF-8".to_string())?;
        Ok(Corpus {
            token_file: None,
            text: Some(text),
        })
    }

    /// The text of a text corpus (a corpus is text or a token file, checked in `load`).
    pub(super) fn text(&self) -> &str {
        self.text
            .as_deref()
            .expect("a corpus is text or a token file")
    }

    /// The corpus as token ids under `tok`; a token file must have been made for its vocabulary.
    pub(super) fn source(
        &mut self,
        tok: &ByteBpeTokenizer,
    ) -> Result<Box<dyn TokenSource>, String> {
        let Some(tf) = self.token_file.take() else {
            return Ok(Box::new(tok.encode_docs(self.text())));
        };
        if tf.vocab_size() != tok.vocab_size() {
            return Err(format!(
                "the token file was made for a vocabulary of {} but the tokenizer has {}",
                tf.vocab_size(),
                tok.vocab_size()
            ));
        }
        Ok(Box::new(tf))
    }
}

/// `--resume`: a checkpoint file to continue from.
pub(super) fn load_resume(a: &Args) -> Result<Option<Checkpoint>, String> {
    a.get("--resume")
        .map(|p| {
            std::fs::read(p)
                .map_err(|e| format!("{p}: {e}"))
                .and_then(|b| from_bytes(b).map_err(|e| format!("{p}: {e}")))
        })
        .transpose()
}

/// `--val`: hold out the tail of the corpus; returns the training length and the held-out part.
pub(super) fn split_validation(
    a: &Args,
    source: &dyn TokenSource,
) -> Result<(usize, Option<Validation>), String> {
    let len = source.len();
    let Some(f) = a.get("--val") else {
        return Ok((len, None));
    };
    let f: f64 = f
        .parse()
        .map_err(|_| format!("--val: '{f}' is not a number"))?;
    if !(0.0..0.9).contains(&f) || f == 0.0 {
        return Err("--val must be in (0, 0.9)".into());
    }
    let cut = len - ((len as f64 * f) as usize).max(2);
    let tokens = source.window(cut, len - cut)?;
    Ok((
        cut,
        Some(Validation {
            tokens,
            patience: a.num("--patience", 0)?,
        }),
    ))
}
