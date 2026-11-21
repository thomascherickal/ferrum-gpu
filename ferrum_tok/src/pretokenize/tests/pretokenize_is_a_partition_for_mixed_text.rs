use super::*;

#[test]
fn pretokenize_is_a_partition_for_mixed_text() {
    let text = "Héllo, wörld!! It's 3.14 — ok?\n\n\tTab\u{a0}end  \r\n日本 😀 'x'  ";
    let chunks = pt(text);
    assert!(chunks.len() > 15);
    let tok = ByteBpeTokenizer::train(&text.repeat(5), 300).unwrap();
    assert_eq!(tok.decode(&tok.encode(text)), text);
}
