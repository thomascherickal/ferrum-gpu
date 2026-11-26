use crate::ByteBpeTokenizer;

#[test]
fn decoding_skips_ids_with_no_vocabulary_entry() {
    let tok = ByteBpeTokenizer::byte_level();
    assert_eq!(tok.decode(&[104, 1_000_000, 105]), "hi");
}
