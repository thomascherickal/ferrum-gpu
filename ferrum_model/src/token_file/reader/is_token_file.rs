use super::*;

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
    pub(in crate::token_file) file: Mutex<File>,
    pub(in crate::token_file) data_start: u64,
    pub(in crate::token_file) width: usize,
    pub(in crate::token_file) tokens: usize,
    pub(in crate::token_file) vocab_size: usize,
    pub(in crate::token_file) documents: u64,
    pub(in crate::token_file) eos_id: Option<u32>,
    pub(in crate::token_file) tokenizer_sha256: String,
}

pub(super) fn num(g: &Gguf, path: &str, key: &str) -> Result<usize, String> {
    g.uint(&format!("ferrum.tokens.{key}"))
        .map_err(|_| format!("{path}: key 'ferrum.tokens.{key}' missing"))
}
