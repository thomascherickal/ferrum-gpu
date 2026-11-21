use super::*;

impl Gguf {
    /// Parse a GGUF v3 file.
    pub fn parse(bytes: Vec<u8>) -> Result<Gguf, GgufError> {
        let total = bytes.len() as u64;
        Gguf::parse_prefix(bytes, total)
    }
}
