//! One-pass atomic writer of a token file.

use super::{element_type, TokenFileMeta, TokenFileStats, TENSOR, TOKEN_FILE_ARCH, VERSION};
use ferrum_gguf::{GgufBuilder, Value, GGML_I32};
use std::io::{BufWriter, Seek, SeekFrom, Write};

/// The GGUF header (metadata, directory, alignment padding) for `n` tokens in `docs` documents.
/// Every count is fixed-width, so the length does not depend on `n` or `docs`: the writer emits a
/// placeholder first and rewrites it in place when the counts are known.
pub fn header_bytes(meta: &TokenFileMeta, n: u64, docs: u64) -> Vec<u8> {
    let (ty, _) = element_type(meta.vocab_size);
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(TOKEN_FILE_ARCH.into()));
    b.meta("ferrum.tokens.version", Value::U32(VERSION));
    b.meta(
        "ferrum.tokens.vocab_size",
        Value::U32(meta.vocab_size as u32),
    );
    b.meta("ferrum.tokens.token_count", Value::U64(n));
    b.meta("ferrum.tokens.document_count", Value::U64(docs));
    b.meta(
        "ferrum.tokens.tokenizer_sha256",
        Value::Str(meta.tokenizer_sha256.clone()),
    );
    if let Some(eos) = meta.eos_id {
        b.meta("ferrum.tokens.eos_id", Value::U32(eos));
    }
    // a tensor that claims `n` ids but carries no data serialises to exactly the header
    b.tensor_raw(TENSOR, &[n], ty, Vec::new());
    b.into_bytes()
}

/// Receives the ids of a token file being written (see [`write`]).
pub struct TokenSink<'a> {
    w: BufWriter<&'a mut std::fs::File>,
    vocab: usize,
    eos: Option<u32>,
    wide: bool,
    tokens: u64,
    eos_seen: u64,
}

impl TokenSink<'_> {
    /// Ids pushed so far.
    pub fn len(&self) -> u64 {
        self.tokens
    }

    /// True when no id was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Append `ids`; one outside the vocabulary is an error.
    pub fn push(&mut self, ids: &[u32]) -> Result<(), String> {
        for &id in ids {
            if id as usize >= self.vocab {
                return Err(format!(
                    "token {} is id {id}, outside the vocabulary of {}",
                    self.tokens, self.vocab
                ));
            }
            self.eos_seen += u64::from(self.eos == Some(id));
            let r = if self.wide {
                self.w.write_all(&id.to_le_bytes())
            } else {
                self.w.write_all(&(id as i16).to_le_bytes())
            };
            r.map_err(|e| e.to_string())?;
            self.tokens += 1;
        }
        Ok(())
    }
}

/// Write a token file to `path` in one pass with O(1) memory: `produce` pushes ids into the
/// [`TokenSink`] (a placeholder header, then the ids through a 1 MiB buffer, then the header
/// rewritten with the real counts). The file appears at `path` only when complete (temp file and
/// rename); an error from `produce` leaves any existing file untouched.
///
/// Ceiling: a `u32` vocabulary and a `u64` token count; 2 bytes per token while the vocabulary is
/// at most 32768, else 4.
pub fn write(
    path: &str,
    meta: &TokenFileMeta,
    produce: impl FnOnce(&mut TokenSink) -> Result<(), String>,
) -> Result<TokenFileStats, String> {
    if meta.vocab_size == 0 || meta.vocab_size > i32::MAX as usize {
        return Err(format!("vocab_size {} out of range", meta.vocab_size));
    }
    if meta.eos_id.is_some_and(|e| e as usize >= meta.vocab_size) {
        return Err("eos_id is outside the vocabulary".into());
    }
    let (ty, width) = element_type(meta.vocab_size);
    let tmp = format!("{path}.part");
    let io = |e: std::io::Error| format!("{path}: {e}");
    let mut f = std::fs::File::create(&tmp).map_err(io)?;
    let outcome = (|| -> Result<TokenFileStats, String> {
        let placeholder = header_bytes(meta, 0, 0);
        f.write_all(&placeholder).map_err(io)?;
        let mut sink = TokenSink {
            w: BufWriter::with_capacity(1 << 20, &mut f),
            vocab: meta.vocab_size,
            eos: meta.eos_id,
            wide: ty == GGML_I32,
            tokens: 0,
            eos_seen: 0,
        };
        produce(&mut sink)?;
        sink.w.flush().map_err(io)?;
        let (n, docs) = (
            sink.tokens,
            if sink.tokens == 0 {
                0
            } else {
                sink.eos_seen + 1
            },
        );
        drop(sink);
        let header = header_bytes(meta, n, docs);
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.write_all(&header).map_err(io)?;
        f.sync_all().map_err(io)?;
        Ok(TokenFileStats {
            tokens: n,
            documents: docs,
            bytes: header.len() as u64 + n * width as u64,
        })
    })();
    drop(f);
    match outcome {
        Ok(stats) => {
            std::fs::rename(&tmp, path).map_err(io)?;
            Ok(stats)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}
