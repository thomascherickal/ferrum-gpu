use super::*;

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
