use super::*;

#[test]
fn a_key_that_is_not_utf8_is_rejected() {
    let mut b = GgufBuilder::new();
    b.meta("a", Value::U8(1));
    let mut bytes = b.into_bytes();
    bytes[24 + 8] = 0xff; // the one key byte, after the header and the 8-byte length
    assert_eq!(
        Gguf::parse(bytes).unwrap_err().0,
        "string is not valid UTF-8"
    );
}
