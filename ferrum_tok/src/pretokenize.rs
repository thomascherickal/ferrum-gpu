// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/pretokenize.rs at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! GPT-2 style pre-tokenization and the byte-to-unicode table.

/// GPT-2 byte → unicode char mapping (OpenAI `bytes_to_unicode`, llama.cpp
/// `unicode_byte_to_utf8_map`): printable Latin-1 bytes map to themselves, the
/// other 68 bytes to U+0100.. in byte order.
pub fn byte_to_unicode(b: u8) -> char {
    let printable = |x: u32| (0x21..=0x7E).contains(&x) || (0xA1..=0xAC).contains(&x) || x >= 0xAE;
    let b32 = u32::from(b);
    if printable(b32) {
        return char::from(b);
    }
    let n = (0..b32).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).expect("256 + n < 324 is a valid char")
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum CharClass {
    Letter,
    Number,
    Space,
    Other,
}

fn char_class(c: char) -> CharClass {
    use unicode_general_category::{get_general_category as cat, GeneralCategory as G};
    if c.is_whitespace() {
        return CharClass::Space;
    }
    match cat(c) {
        G::UppercaseLetter
        | G::LowercaseLetter
        | G::TitlecaseLetter
        | G::ModifierLetter
        | G::OtherLetter => CharClass::Letter,
        G::DecimalNumber | G::LetterNumber | G::OtherNumber => CharClass::Number,
        _ => CharClass::Other,
    }
}

/// Pre-tokenize `text` exactly like llama.cpp's `gpt-2` pre-tokenizer
/// (`unicode_regex_split_custom_gpt2`, b11521):
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
/// The chunks are a partition of `text`, so encode/decode still round-trips and
/// learned merges never span chunks.
pub(super) fn pretokenize(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |p: usize| cs.get(p).map(|&(_, c)| c);
    let class = |p: usize| at(p).map(char_class);
    let mut chunks = Vec::new();
    let mut prev = 0; // char index where the current chunk starts
    let mut pos = 0;
    while pos < n {
        let c = cs[pos].1;
        let mut end = None;
        // Contractions.
        if c == '\'' && pos + 1 < n {
            let (c1, c2) = (cs[pos + 1].1, at(pos + 2));
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                end = Some(pos + 2);
            } else if matches!(
                (c1, c2),
                ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))
            ) {
                end = Some(pos + 3);
            }
        }
        if end.is_none() {
            // ` ?` then a run of letters / numbers / other.
            let lead = usize::from(c == ' ');
            let cl = class(pos + lead);
            if matches!(
                cl,
                Some(CharClass::Letter | CharClass::Number | CharClass::Other)
            ) {
                let mut e = pos + lead;
                while class(e) == cl {
                    e += 1;
                }
                end = Some(e);
            }
        }
        let end = end.unwrap_or_else(|| {
            // Whitespace run: leave the last one for the next word (`\s+(?!\S)`)
            // unless the run reaches the end of the text.
            let mut e = pos;
            while class(e) == Some(CharClass::Space) {
                e += 1;
            }
            if e - pos > 1 && e < n {
                e - 1
            } else {
                e.max(pos + 1)
            }
        });
        let byte = |i: usize| cs.get(i).map_or(text.len(), |&(b, _)| b);
        chunks.push(&text[byte(prev)..byte(end)]);
        prev = end;
        pos = end;
    }
    chunks
}
#[cfg(test)]
mod tests;
