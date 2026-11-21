//! The tokenizer stored in an imported Llama/Qwen GGUF (`tokenizer.ggml.*`), so those models can be
//! driven with text. Two families: byte-level BPE (`gpt2`/`bpe`: Llama-3, Qwen2) and SentencePiece
//! (`llama`/`spm`).
//!
//! Provenance: ported from ferrum-cpu `ferrum_core/src/gguf_tokenizer.rs` (commit 56a9d3a) with the
//! metadata access rewritten for our `ferrum_gguf`. Limits, unchanged from the original: the BPE
//! pre-tokenizer is a dependency-free approximation of the GPT-2/Qwen regex (it can differ on
//! exotic whitespace), and SPM encoding is a greedy longest match, not SentencePiece's Viterbi
//! search. `decode` is exact for both. Pass token ids (`--ids`) where exactness matters.

use ferrum_gguf::{Gguf, Value};
use std::collections::HashMap;

/// SentencePiece's visible space marker (U+2581, "▁").
const SPM_SPACE: char = '\u{2581}';

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
    model: TokModel,
    /// id → token piece, exactly as stored in the GGUF.
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// BPE merge ranks keyed by the `left\0right` piece pair (lower = earlier).
    /// A single joined `String` key (with a NUL separator, which never occurs in
    /// a GPT-2 byte-encoded piece) lets the hot merge scan look up pairs through
    /// a reused buffer with **zero per-pair allocation** (`get(&str)` via
    /// `Borrow`), instead of cloning both pieces into a `(String, String)` key.
    merges: HashMap<String, u32>,
    /// Longest vocabulary token in **chars** — bounds the SPM longest-match
    /// window so `encode_spm` is O(n·max_token_len), not O(n³).
    max_token_len: usize,
    /// SPM `<0xXX>` byte-fallback tokens: raw byte → token id.
    byte_to_id: HashMap<u8, u32>,
    /// GPT-2 byte ↔ printable-unicode tables (BPE only).
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bos: Option<u32>,
    eos: Option<u32>,
    /// `tokenizer.ggml.add_bos_token` (llama.cpp default: true for SPM, false for BPE).
    add_bos: bool,
}

/// Build GPT-2's byte→unicode table (and its inverse). Printable byte ranges map
/// to themselves; the rest map to code points ≥ 256, so every byte is a
/// printable, single-`char` token — the trick that makes byte-level BPE total.
fn gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
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
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

fn string_array(g: &Gguf, key: &str) -> Option<Vec<String>> {
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

impl GgufTokenizer {
    /// Reconstruct the tokenizer from a parsed GGUF's metadata. Errors if the
    /// file declares no tokenizer or an unsupported family.
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let model_str = g
            .string("tokenizer.ggml.model")
            .map_err(|_| "GGUF has no tokenizer.ggml.model metadata".to_string())?;
        let model = match model_str {
            "gpt2" | "bpe" => TokModel::Bpe,
            "llama" | "spm" => TokModel::Spm,
            other => {
                return Err(format!(
                    "unsupported GGUF tokenizer model '{other}' (have: gpt2/bpe, llama/spm)"
                ))
            }
        };

        let tokens =
            string_array(g, "tokenizer.ggml.tokens").ok_or("GGUF missing tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err("GGUF tokenizer.ggml.tokens is empty".into());
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = HashMap::new();
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.entry(tok.clone()).or_insert(id as u32);
            if let Some(b) = parse_byte_token(tok) {
                byte_to_id.insert(b, id as u32);
            }
        }

        // BPE merges: "left right" piece pairs, in priority order. Stored under a
        // `left\0right` joined key (see the `merges` field docs).
        let mut merges = HashMap::new();
        if let Some(pairs) = string_array(g, "tokenizer.ggml.merges") {
            for (rank, m) in pairs.iter().enumerate() {
                if let Some(sp) = m.find(' ') {
                    let key = format!("{}\u{0}{}", &m[..sp], &m[sp + 1..]);
                    merges.entry(key).or_insert(rank as u32);
                }
            }
        }

        // Longest token (in chars) for the SPM longest-match bound.
        let max_token_len = tokens
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);

        let bos = g.uint("tokenizer.ggml.bos_token_id").ok().map(|u| u as u32);
        let eos = g.uint("tokenizer.ggml.eos_token_id").ok().map(|u| u as u32);

        let add_bos = match g.get("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(b)) => *b,
            // llama.cpp's per-pre-tokenizer defaults (llama-vocab.cpp, b11521)
            _ => {
                model == TokModel::Spm
                    || matches!(
                        g.string("tokenizer.ggml.pre").ok(),
                        Some(
                            "llama3"
                                | "llama-v3"
                                | "llama-bpe"
                                | "falcon3"
                                | "falcon-h1"
                                | "pixtral"
                                | "midm-2.0"
                                | "lfm2"
                                | "jina-v5-nano"
                                | "tekken"
                                | "chameleon"
                        )
                    )
            }
        };
        let (byte_encoder, byte_decoder) = gpt2_byte_tables();
        Ok(Self {
            model,
            tokens,
            token_to_id,
            merges,
            max_token_len,
            byte_to_id,
            byte_encoder,
            byte_decoder,
            bos,
            eos,
            add_bos,
        })
    }

    /// The tokenizer family declared by the file (`tokenizer.ggml.model`).
    pub fn model(&self) -> TokModel {
        self.model
    }
    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    /// The beginning-of-sequence token id, if the file defines one.
    pub fn bos(&self) -> Option<usize> {
        self.bos.map(|x| x as usize)
    }
    /// The BOS id to prepend to a prompt: `bos()` only when the file asks for it
    /// (`add_bos_token`, defaulting like llama.cpp). Prepending an unwanted BOS changes
    /// the model's output (e.g. SmolLM2 has a BOS id but `add_bos_token = false`).
    pub fn prefix_bos(&self) -> Option<usize> {
        self.bos().filter(|_| self.add_bos)
    }
    /// The end-of-sequence token id, if the file defines one.
    pub fn eos(&self) -> Option<usize> {
        self.eos.map(|x| x as usize)
    }

    /// Exact-piece vocabulary lookup (used to map chat special tokens to ids).
    pub fn token_id(&self, piece: &str) -> Option<usize> {
        self.token_to_id.get(piece).map(|&id| id as usize)
    }

    /// Encode text to token IDs. BPE is exact (given the approximate
    /// pre-tokenizer); SPM is a greedy longest-match approximation with byte
    /// fallback. See the module docs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.model {
            TokModel::Bpe => self.encode_bpe(text),
            TokModel::Spm => self.encode_spm(text),
        }
    }

    /// Decode token IDs back to text. Exact for both families.
    pub fn decode(&self, ids: &[usize]) -> String {
        match self.model {
            TokModel::Bpe => self.decode_bpe(ids),
            TokModel::Spm => self.decode_spm(ids),
        }
    }

    // ── BPE ───────────────────────────────────────────────────────────────────

    fn encode_bpe(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            // Byte-level map: each byte → one printable GPT-2 char (a "piece").
            let mut pieces: Vec<String> = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();
            // Greedy lowest-rank merges until none apply (standard BPE). `key` is
            // reused across every pair lookup so the scan allocates nothing.
            let mut key = String::new();
            while pieces.len() > 1 {
                let mut best: Option<(usize, u32)> = None;
                for i in 0..pieces.len() - 1 {
                    key.clear();
                    key.push_str(&pieces[i]);
                    key.push('\u{0}');
                    key.push_str(&pieces[i + 1]);
                    if let Some(&r) = self.merges.get(key.as_str()) {
                        let better = match best {
                            None => true,
                            Some((_, br)) => r < br,
                        };
                        if better {
                            best = Some((i, r));
                        }
                    }
                }
                let Some((i, _)) = best else { break };
                let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                pieces[i] = merged;
                pieces.remove(i + 1);
            }
            for p in pieces {
                if let Some(&id) = self.token_to_id.get(&p) {
                    out.push(id as usize);
                } else {
                    // Unmergeable leftover: emit its constituent byte tokens,
                    // which always exist in a byte-level vocab.
                    for ch in p.chars() {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            out.push(id as usize);
                        }
                    }
                }
            }
        }
        out
    }

    fn decode_bpe(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            for ch in tok.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    // A non-byte char (e.g. an added special token's literal
                    // text) — emit its own UTF-8.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ── SPM ───────────────────────────────────────────────────────────────────

    fn encode_spm(&self, text: &str) -> Vec<usize> {
        // SentencePiece prepends a dummy space and renders spaces as ▁.
        let normalized: Vec<char> = format!(" {text}")
            .replace(' ', &SPM_SPACE.to_string())
            .chars()
            .collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < normalized.len() {
            // Longest token in the vocab starting at i. The window is capped at
            // the longest vocab token, so this is O(max_token_len) per position
            // (not O(remaining-length)) — without the cap, encoding a long corpus
            // is O(n³) and effectively hangs.
            let mut hit = None;
            let mut j = (i + self.max_token_len).min(normalized.len());
            while j > i {
                let cand: String = normalized[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&cand) {
                    hit = Some((id, j));
                    break;
                }
                j -= 1;
            }
            match hit {
                Some((id, j)) => {
                    out.push(id as usize);
                    i = j;
                }
                None => {
                    // Byte fallback: emit each UTF-8 byte's <0xXX> token.
                    let mut buf = [0u8; 4];
                    for &b in normalized[i].encode_utf8(&mut buf).as_bytes() {
                        if let Some(&id) = self.byte_to_id.get(&b) {
                            out.push(id as usize);
                        }
                    }
                    i += 1;
                }
            }
        }
        out
    }

    fn decode_spm(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            if let Some(b) = parse_byte_token(tok) {
                bytes.push(b);
                continue;
            }
            for ch in tok.chars() {
                if ch == SPM_SPACE {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        // Drop the leading space SentencePiece's dummy prefix introduced.
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}

/// Dependency-free approximation of GPT-2 pre-tokenization: split into runs of
/// letters / digits / symbols, with a single leading space attached to the
/// following run (so the space becomes the BPE `Ġ` marker). Whitespace runs are
/// emitted as their own chunks.
fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if chars[i] == ' ' {
            i += 1; // attach one leading space to the following run
        }
        if i >= chars.len() {
            chunks.push(chars[start..].iter().collect());
            break;
        }
        let c = chars[i];
        if c.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
        } else if c.is_numeric() {
            while i < chars.len() && chars[i].is_numeric() {
                i += 1;
            }
        } else if c.is_whitespace() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        } else {
            while i < chars.len() && !chars[i].is_alphanumeric() && !chars[i].is_whitespace() {
                i += 1;
            }
        }
        chunks.push(chars[start..i].iter().collect());
    }
    chunks
}
#[cfg(test)]
mod tests;
