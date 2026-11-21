use super::*;

#[test]
fn punctuation_folds_and_blank_runs_collapse() {
    let got = clean(
        "\u{201C}a\u{201D} \u{2018}b\u{2019}\u{2014}c\u{2026}\r\n\n\n\nd  \n",
        false,
    );
    assert_eq!(got, "\"a\" 'b'-c...\n\nd\n");
    assert_eq!(clean("ABC\u{200B}\u{7}d", true), "abcd\n");
    assert_eq!(clean("\n\n", false), "");
}

#[test]
fn no_break_spaces_and_bullets_are_folded() {
    assert_eq!(
        clean("a\u{00A0}b\u{2007}c\u{202F}d \u{2022} e", false),
        "a b c d * e\n"
    );
}

#[test]
fn invisible_characters_and_spaces() {
    assert_eq!(strip_invisible("\u{FEFF}a\u{200D}b\r\u{1}\tc\n"), "ab\tc\n");
    assert_eq!(squeeze_spaces("a  \t b\n c"), "a b\n c");
    assert_eq!(nfkc("\u{FB01}ne \u{FF21}"), "fine A");
}

#[test]
fn gutenberg_keeps_every_body_and_falls_back_to_the_whole_text() {
    let raw = "junk\n*** START OF A ***\none\n*** END OF A ***\nmid\n***START OF B***\ntwo\n***END OF B***\ntail";
    assert_eq!(gutenberg_bodies(raw), vec!["one\n", "two\n"]);
    assert_eq!(gutenberg_bodies("no markers"), vec!["no markers"]);
    assert_eq!(
        gutenberg_bodies("*** START OF A ***\nopen"),
        vec!["*** START OF A ***\nopen"]
    );
    assert_eq!(gutenberg_bodies("*** START OF A"), vec!["*** START OF A"]);
    assert_eq!(
        gutenberg_bodies("**** START OF A ***\nx\n*** END OF"),
        vec!["x\n"]
    );
}
