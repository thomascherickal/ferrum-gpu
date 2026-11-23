use super::*;

#[test]
fn bytes_that_are_not_utf8_inside_a_string_are_an_error() {
    // `Json::parse` takes a `&str`, which is always valid UTF-8; the string reader works on bytes.
    let mut reader = value::P {
        b: b"\"a\xff\"",
        i: 0,
    };
    assert_eq!(reader.string().unwrap_err(), "invalid UTF-8 in string");
}
