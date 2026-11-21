use super::*;

#[test]
fn forward_rejects_bad_input() {
    let w = Gpt2::random(tiny(), 3);
    assert!(w
        .forward_logits(&[])
        .unwrap_err()
        .contains("sequence length"));
    assert!(w
        .forward_logits(&[0; 9])
        .unwrap_err()
        .contains("sequence length"));
    assert!(w
        .forward_logits(&[11])
        .unwrap_err()
        .contains("out of range"));
}
