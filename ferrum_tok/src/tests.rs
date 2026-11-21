use super::ByteBpeTokenizer;

#[test]
fn byte_level_roundtrip() {
    let tok = ByteBpeTokenizer::byte_level();
    let text = "hello 🌸 мир\n";
    let ids = tok.encode(text);
    assert_eq!(ids.len(), text.len()); // one token per byte
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn token_bytes_lookup() {
    let tok = ByteBpeTokenizer::byte_level();
    assert_eq!(tok.token_bytes(b'a' as usize), Some(&b"a"[..]));
    assert_eq!(tok.token_bytes(999), None);
}
