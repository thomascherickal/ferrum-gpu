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
pub(crate) enum CharClass {
    Letter,
    Number,
    Space,
    Other,
}

pub(super) fn char_class(c: char) -> CharClass {
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
