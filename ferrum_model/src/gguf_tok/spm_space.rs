use super::*;

/// SentencePiece's visible space marker (U+2581, "▁").
pub(super) const SPM_SPACE: char = '\u{2581}';

/// Which tokenizer family a GGUF declares in `tokenizer.ggml.model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokModel {
    /// Byte-level BPE (`gpt2` / `bpe`): Llama-3, Qwen2, GPT-2, …
    Bpe,
    /// SentencePiece unigram (`llama` / `spm`): Llama-1/2, …
    Spm,
}

/// A tokenizer reconstructed from a GGUF metadata table.
pub struct GgufTokenizer {
    pub(super) model: TokModel,
    /// id → token piece, exactly as stored in the GGUF.
    pub(super) tokens: Vec<String>,
    pub(super) token_to_id: HashMap<String, u32>,
    /// BPE merge ranks keyed by the `left\0right` piece pair (lower = earlier).
    /// A single joined `String` key (with a NUL separator, which never occurs in
    /// a GPT-2 byte-encoded piece) lets the hot merge scan look up pairs through
    /// a reused buffer with **zero per-pair allocation** (`get(&str)` via
    /// `Borrow`), instead of cloning both pieces into a `(String, String)` key.
    pub(super) merges: HashMap<String, u32>,
    /// Longest vocabulary token in **chars** — bounds the SPM longest-match
    /// window so `encode_spm` is O(n·max_token_len), not O(n³).
    pub(super) max_token_len: usize,
    /// SPM `<0xXX>` byte-fallback tokens: raw byte → token id.
    pub(super) byte_to_id: HashMap<u8, u32>,
    /// GPT-2 byte ↔ printable-unicode tables (BPE only).
    pub(super) byte_encoder: [char; 256],
    pub(super) byte_decoder: HashMap<char, u8>,
    pub(super) bos: Option<u32>,
    pub(super) eos: Option<u32>,
    /// `tokenizer.ggml.add_bos_token` (llama.cpp default: true for SPM, false for BPE).
    pub(super) add_bos: bool,
}

/// Build GPT-2's byte→unicode table (and its inverse). Printable byte ranges map
/// to themselves; the rest map to code points ≥ 256, so every byte is a
/// printable, single-`char` token — the trick that makes byte-level BPE total.
pub(super) fn gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    for b in 0xA1u8..=0xAC {
        bs.push(b);
    }
    for b in 0xAEu8..=0xFF {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u16..256 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = ['\0'; 256];
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        enc[b as usize] = char::from_u32(c).unwrap_or('\u{FFFD}');
    }
    let mut dec = HashMap::with_capacity(256);
    for (b, &c) in enc.iter().enumerate() {
        dec.insert(c, b as u8);
    }
    (enc, dec)
}

/// Parse a SentencePiece byte-fallback token `<0xHH>` into its raw byte.
pub(super) fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

pub(super) fn string_array(g: &Gguf, key: &str) -> Option<Vec<String>> {
    match g.get(key) {
        Some(Value::Arr(items)) => Some(
            items
                .iter()
                .map(|v| match v {
                    Value::Str(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}
