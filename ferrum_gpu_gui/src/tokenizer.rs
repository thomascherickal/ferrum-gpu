//! The byte-level tokenizer the GUI uses until the real BPE tokenizer arrives (Phase 6): one token
//! per byte, vocabulary 256. Decoding is lossy only where bytes are not valid UTF-8.

/// Vocabulary size.
pub const VOCAB: usize = 256;

/// Text to token ids (its UTF-8 bytes).
pub fn encode(text: &str) -> Vec<usize> {
    text.bytes().map(usize::from).collect()
}

/// Token ids to text; ids of 256 or more are refused, invalid UTF-8 becomes U+FFFD.
pub fn decode(ids: &[usize]) -> Result<String, String> {
    let bytes: Result<Vec<u8>, String> = ids
        .iter()
        .map(|&i| u8::try_from(i).map_err(|_| format!("token {i} is not a byte")))
        .collect();
    Ok(String::from_utf8_lossy(&bytes?).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_text_including_multibyte() {
        for s in ["", "hello", "héllo wörld", "日本語 😀", "tab\tnew\nline"] {
            assert_eq!(decode(&encode(s)).unwrap(), s);
        }
        assert_eq!(encode("A"), vec![65]);
        assert_eq!(encode("é"), vec![0xC3, 0xA9]);
    }

    #[test]
    fn invalid_ids_are_refused_and_broken_utf8_is_replaced() {
        assert!(decode(&[65, 256]).unwrap_err().contains("not a byte"));
        assert_eq!(decode(&[0xFF, 65]).unwrap(), "\u{FFFD}A");
        assert_eq!(VOCAB, 256);
    }
}
