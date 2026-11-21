use super::*;

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
