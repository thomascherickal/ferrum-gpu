use super::*;

impl TokenFile {
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
