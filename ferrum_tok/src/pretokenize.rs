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
mod tests {
    use super::{byte_to_unicode, pretokenize};
    use crate::ByteBpeTokenizer;

    #[test]
    fn pretokenize_partitions_text() {
        assert_eq!(pretokenize("low lower"), vec!["low", " lower"]);
        assert_eq!(pretokenize("a\n b"), vec!["a", "\n", " b"]);
        assert_eq!(pretokenize(""), Vec::<&str>::new());
        assert_eq!(pretokenize("solid"), vec!["solid"]);
        // The chunks always reconstruct the original text.
        let s = "the  quick\tbrown\nfox";
        assert_eq!(pretokenize(s).concat(), s);
    }

    #[test]
    fn pretokenize_matches_gpt2_regex() {
        let cases: &[(&str, &[&str])] = &[
            ("Hello world", &["Hello", " world"]),
            (
                "I'm here, it's 2024!",
                &["I", "'m", " here", ",", " it", "'s", " 2024", "!"],
            ),
            ("a  b", &["a", " ", " b"]),
            ("x\n\ny", &["x", "\n", "\n", "y"]),
            ("end   ", &["end", "   "]),
            ("naïve café", &["naïve", " café"]),
            ("<|endoftext|>Once", &["<|", "endoftext", "|>", "Once"]),
        ];
        for (text, want) in cases {
            let got = pretokenize(text);
            assert_eq!(&got, want, "{text:?}");
            assert_eq!(got.concat(), *text);
        }
    }

    #[test]
    fn byte_to_unicode_matches_gpt2() {
        assert_eq!(byte_to_unicode(b'A'), 'A');
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
        assert_eq!(byte_to_unicode(b'\n'), 'Ċ');
        let all: std::collections::HashSet<char> = (0..=255u8).map(byte_to_unicode).collect();
        assert_eq!(all.len(), 256);
    }

    fn pt(text: &str) -> Vec<&str> {
        let got = pretokenize(text);
        assert_eq!(got.concat(), text, "chunks must partition {text:?}");
        assert!(got.iter().all(|c| !c.is_empty()), "empty chunk in {got:?}");
        got
    }

    #[test]
    fn byte_to_unicode_full_table_matches_openai_algorithm() {
        // OpenAI gpt-2 `bytes_to_unicode()`, computed here independently.
        let mut bs: Vec<u32> = (0x21..=0x7E)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        assert_eq!((bs.len(), n), (256, 68));
        for (b, c) in bs.iter().zip(&cs) {
            assert_eq!(byte_to_unicode(*b as u8) as u32, *c, "byte {b:#04x}");
        }
        // spot checks of the published table
        assert_eq!(byte_to_unicode(0x00), '\u{100}');
        assert_eq!(byte_to_unicode(0x20), '\u{120}'); // Ġ
        assert_eq!(byte_to_unicode(0x7F), '\u{121}');
        assert_eq!(byte_to_unicode(0xA0), '\u{142}');
        assert_eq!(byte_to_unicode(0xAD), '\u{143}');
        assert_eq!(byte_to_unicode(0xFF), 'ÿ');
        let uniq: std::collections::HashSet<char> = (0..=255u8).map(byte_to_unicode).collect();
        assert_eq!(uniq.len(), 256, "table is injective");
    }

    #[test]
    fn pretokenize_contractions() {
        assert_eq!(pt("don't"), ["don", "'t"]);
        assert_eq!(
            pt("I've we're they'll he'd she's I'm"),
            ["I", "'ve", " we", "'re", " they", "'ll", " he", "'d", " she", "'s", " I", "'m"]
        );
        // case-sensitive, and incomplete suffixes are plain punctuation
        assert_eq!(pt("'S"), ["'", "S"]);
        assert_eq!(pt("'RE"), ["'", "RE"]);
        assert_eq!(pt("'r"), ["'", "r"]);
        assert_eq!(pt("'l"), ["'", "l"]);
        assert_eq!(pt("'v"), ["'", "v"]);
        assert_eq!(pt("it'"), ["it", "'"]);
        assert_eq!(pt("'re"), ["'re"]);
        assert_eq!(pt("'ll"), ["'ll"]);
        assert_eq!(pt("x'rex"), ["x", "'re", "x"]);
        assert_eq!(pt("''s"), ["''", "s"]); // contraction only at the start of a token
        assert_eq!(pt("'t'd"), ["'t", "'d"]);
        assert_eq!(pt("a 's"), ["a", " '", "s"]); // space-led: ` ?[^\s\p{L}\p{N}]+` wins
    }

    #[test]
    fn pretokenize_whitespace_rules() {
        assert_eq!(pt(""), Vec::<&str>::new());
        assert_eq!(pt(" "), [" "]);
        assert_eq!(pt("   "), ["   "]);
        assert_eq!(pt("\n\n\n"), ["\n\n\n"]);
        assert_eq!(pt("  x"), [" ", " x"]);
        assert_eq!(pt("x \n"), ["x", " \n"]);
        assert_eq!(pt("a\n\n b"), ["a", "\n\n", " b"]);
        assert_eq!(pt("a\tb"), ["a", "\t", "b"]);
        assert_eq!(pt("a \tb"), ["a", " ", "\t", "b"]);
        assert_eq!(pt("a  !!"), ["a", " ", " !!"]);
        assert_eq!(pt("\n word"), ["\n", " word"]);
        assert_eq!(pt("\r\n\r\nx"), ["\r\n\r", "\n", "x"]);
        // Unicode White_Space is whitespace; a lone leading ' ' only attaches to the next word
        assert_eq!(pt("a\u{a0}b"), ["a", "\u{a0}", "b"]);
        assert_eq!(pt("\u{2003}x"), ["\u{2003}", "x"]);
        assert_eq!(pt(" \u{a0}b"), [" ", "\u{a0}", "b"]);
        assert_eq!(pt("a\u{3000}\u{3000}"), ["a", "\u{3000}\u{3000}"]);
    }

    #[test]
    fn pretokenize_char_classes() {
        assert_eq!(pt("a1b2"), ["a", "1", "b", "2"]);
        assert_eq!(pt(" 12ab"), [" 12", "ab"]);
        assert_eq!(pt("x 3.14"), ["x", " 3", ".", "14"]);
        assert_eq!(pt(" !"), [" !"]);
        assert_eq!(pt("?!.. x"), ["?!..", " x"]);
        // No (superscript), Nl (roman numeral) are numbers; Nd in other scripts too
        assert_eq!(pt("a\u{b2}b"), ["a", "\u{b2}", "b"]);
        assert_eq!(pt("x\u{2167}y"), ["x", "\u{2167}", "y"]);
        assert_eq!(pt(" \u{663}\u{664}z"), [" \u{663}\u{664}", "z"]);
        // letters: Lo (CJK/kana), Ll/Lu (Greek), Lm (modifier), Lt (titlecase)
        assert_eq!(pt("日本語 テスト"), ["日本語", " テスト"]);
        assert_eq!(pt("αβγ Δ"), ["αβγ", " Δ"]);
        assert_eq!(pt("a\u{2b0}b"), ["a\u{2b0}b"]);
        assert_eq!(pt("\u{1c5}x"), ["\u{1c5}x"]);
        // combining marks are not letters: they split off as punctuation-like runs
        assert_eq!(pt("e\u{301}x"), ["e", "\u{301}", "x"]);
        // symbols, emoji, ZWJ (Cf), controls and private use are all `[^\s\p{L}\p{N}]`
        assert_eq!(pt("😀😀 😀"), ["😀😀", " 😀"]);
        assert_eq!(pt("a\u{200d}b"), ["a", "\u{200d}", "b"]);
        assert_eq!(pt("\u{1}\u{2}x"), ["\u{1}\u{2}", "x"]);
        assert_eq!(pt("\u{e000}\u{e001}"), ["\u{e000}\u{e001}"]);
        // Unassigned (Cn) code points are `[^\s\p{L}\p{N}]` too: llama.cpp b11521 gives
        // them the UNDEFINED flag, which is non-zero, so they join "other" runs
        // (checked against its unicode.cpp and, for ids, by the llama-tokenize
        // parity test in tests/tests/test_runtime_compat.rs).
        assert_eq!(pt("a\u{378}\u{379}!b"), ["a", "\u{378}\u{379}!", "b"]);
        assert_eq!(pt("a \u{378} b"), ["a", " \u{378}", " b"]);
    }

    #[test]
    fn pretokenize_unassigned_codepoints_lump_with_other_like_llama_cpp() {
        // llama.cpp b11521's table marks Cn as UNDEFINED (a nonzero flag), so its
        // `[^\s\p{L}\p{N}]+` branch accepts them: they join the punctuation run
        // rather than becoming one chunk each.
        assert!(char::from_u32(0x378).is_some());
        assert_eq!(pt("a\u{378}\u{379}b"), ["a", "\u{378}\u{379}", "b"]);
        assert_eq!(pt(" \u{378}z"), [" \u{378}", "z"]);
        assert_eq!(pt("!\u{378}?"), ["!\u{378}?"]);
    }

    #[test]
    fn pretokenize_is_a_partition_for_mixed_text() {
        let text = "Héllo, wörld!! It's 3.14 — ok?\n\n\tTab\u{a0}end  \r\n日本 😀 'x'  ";
        let chunks = pt(text);
        assert!(chunks.len() > 15);
        let tok = ByteBpeTokenizer::train(&text.repeat(5), 300).unwrap();
        assert_eq!(tok.decode(&tok.encode(text)), text);
    }
}
