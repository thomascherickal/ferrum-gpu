use super::*;

/// Why a document of `n` characters is dropped for its length, if it is.
pub(super) fn length_reject(o: &PrepareOptions, n: usize) -> Option<&'static str> {
    if n == 0 {
        Some("empty")
    } else if n < o.min_chars {
        Some("too_short")
    } else if o.max_chars > 0 && n > o.max_chars {
        Some("too_long")
    } else {
        None
    }
}

/// The first 16 bytes of the document's sha256: the key duplicates are found by.
pub(super) fn digest(doc: &str) -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(doc.as_bytes());
    let d = h.digest();
    std::array::from_fn(|i| d[i])
}
