use super::*;
use crate::path_context::at;

/// Where the tensor data starts in the file.
pub(in crate::token_file) fn tensor_start(
    data_offset: usize,
    offset: u64,
    path: &str,
) -> Result<u64, String> {
    (data_offset as u64)
        .checked_add(offset)
        .ok_or_else(|| format!("{path}: tensor offset overflows"))
}

impl TokenFile {
    /// Open and validate `path`: GGUF header, architecture, version, metadata, and one 1-D integer
    /// `tokens` tensor whose length matches `token_count` and lies inside the file (a truncated
    /// file fails here). The ids are range-checked as they are read.
    pub fn open(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let total = at(file.metadata(), path)?.len();
        let mut head = Vec::new();
        at(file.by_ref().take(1 << 20).read_to_end(&mut head), path)?;
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
        let data_start = tensor_start(g.data_offset(), t.offset, path)?;
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
}
