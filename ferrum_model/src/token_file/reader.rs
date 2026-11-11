//! Streaming reader of a token file: the header is parsed, the ids are read window by window.

use super::{element_type, TENSOR, TOKEN_FILE_ARCH, VERSION};
use ferrum_gguf::{Gguf, Value};
use ferrum_ref::trainer::TokenSource;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

/// Does `path` start like a GGUF file (magic and version 2 or 3)? Lets `train` tell a token file
/// from a text corpus by content: a text file cannot match, since bytes 4..8 would have to be
/// the control characters `02 00 00 00`.
pub fn is_token_file(path: &str) -> bool {
    let mut head = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok()
        && head.starts_with(b"GGUF")
        && matches!(
            u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
            2 | 3
        )
}

/// An open token file. It holds the file handle, never the ids.
#[derive(Debug)]
pub struct TokenFile {
    file: Mutex<File>,
    data_start: u64,
    width: usize,
    tokens: usize,
    vocab_size: usize,
    documents: u64,
    eos_id: Option<u32>,
    tokenizer_sha256: String,
}

fn num(g: &Gguf, path: &str, key: &str) -> Result<usize, String> {
    g.uint(&format!("ferrum.tokens.{key}"))
        .map_err(|_| format!("{path}: key 'ferrum.tokens.{key}' missing"))
}

impl TokenFile {
    /// Open and validate `path`: GGUF header, architecture, version, metadata, and one 1-D integer
    /// `tokens` tensor whose length matches `token_count` and lies inside the file (a truncated
    /// file fails here). The ids are range-checked as they are read.
    pub fn open(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let total = file.metadata().map_err(|e| format!("{path}: {e}"))?.len();
        let mut head = Vec::new();
        file.by_ref()
            .take(1 << 20)
            .read_to_end(&mut head)
            .map_err(|e| format!("{path}: {e}"))?;
        let g = Gguf::parse_prefix(head, total).map_err(|e| format!("{path}: {e}"))?;
        match g.string("general.architecture") {
            Ok(TOKEN_FILE_ARCH) => {}
            other => {
                return Err(format!(
                    "{path} is not a {TOKEN_FILE_ARCH} file (general.architecture = {other:?})"
                ))
            }
        }
        if num(&g, path, "version")? != VERSION as usize {
            return Err(format!("{path}: unsupported token file version"));
        }
        let vocab_size = num(&g, path, "vocab_size")?;
        if vocab_size == 0 || vocab_size > i32::MAX as usize {
            return Err(format!("{path}: vocab_size {vocab_size} out of range"));
        }
        let tokens = num(&g, path, "token_count")?;
        let documents = num(&g, path, "document_count")? as u64;
        let eos_id = match g.get("ferrum.tokens.eos_id") {
            None => None,
            Some(_) => Some(
                u32::try_from(num(&g, path, "eos_id")?)
                    .ok()
                    .filter(|&e| (e as usize) < vocab_size)
                    .ok_or_else(|| format!("{path}: eos_id is outside the vocabulary"))?,
            ),
        };
        let tokenizer_sha256 = match g.get("ferrum.tokens.tokenizer_sha256") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(format!(
                    "{path}: key 'ferrum.tokens.tokenizer_sha256' missing"
                ))
            }
        };
        let (ty, width) = element_type(vocab_size);
        let t = match g.tensors.as_slice() {
            [t] if t.name == TENSOR => t,
            _ => return Err(format!("{path}: expected exactly one '{TENSOR}' tensor")),
        };
        if t.ggml_type != ty || t.dims != [tokens as u64] {
            return Err(format!("{path}: tensor '{TENSOR}' must be 1-D [{tokens}] of type {ty} for a vocabulary of {vocab_size}"));
        }
        let data_start = (g.data_offset() as u64)
            .checked_add(t.offset)
            .ok_or_else(|| format!("{path}: tensor offset overflows"))?;
        Ok(TokenFile {
            file: Mutex::new(file),
            data_start,
            width,
            tokens,
            vocab_size,
            documents,
            eos_id,
            tokenizer_sha256,
        })
    }

    /// Vocabulary size the ids were made for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Documents recorded in the header.
    pub fn documents(&self) -> u64 {
        self.documents
    }

    /// The end-of-text id, if the tokenizer had one.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// SHA-256 of the tokenizer file the ids came from.
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Read ids `start..start + len` (all inside the file).
    pub fn read(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        if start.checked_add(len).is_none_or(|e| e > self.tokens) {
            return Err(format!(
                "tokens {start}+{len} are outside the {} in the file",
                self.tokens
            ));
        }
        let mut buf = vec![0u8; len * self.width];
        {
            let mut f = self.file.lock().unwrap_or_else(|e| e.into_inner());
            f.seek(SeekFrom::Start(
                self.data_start + (start * self.width) as u64,
            ))
            .map_err(|e| e.to_string())?;
            f.read_exact(&mut buf)
                .map_err(|e| format!("reading the token file: {e}"))?;
        }
        buf.chunks_exact(self.width)
            .enumerate()
            .map(|(i, c)| {
                let id = if self.width == 2 {
                    i64::from(i16::from_le_bytes([c[0], c[1]]))
                } else {
                    i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                };
                usize::try_from(id)
                    .ok()
                    .filter(|&v| v < self.vocab_size)
                    .ok_or_else(|| {
                        format!(
                            "token {} is id {id}, outside the vocabulary of {}",
                            start + i,
                            self.vocab_size
                        )
                    })
            })
            .collect()
    }
}

impl TokenSource for TokenFile {
    fn len(&self) -> usize {
        self.tokens
    }

    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        self.read(start, len)
    }
}
