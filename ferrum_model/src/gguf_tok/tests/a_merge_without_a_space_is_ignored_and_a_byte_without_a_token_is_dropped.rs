use super::*;

#[test]
fn a_merge_without_a_space_is_ignored_and_a_byte_without_a_token_is_dropped() {
    // "nospace" has no separator, so it is not a merge; the encoding is the plain pieces
    let t = build("gpt2", &["a", "b", "ab"], &["nospace", "a b"], &[]).unwrap();
    assert_eq!(t.encode("ab"), vec![2]);
    // SPM: "z" is not a token and the vocabulary has no <0x7A> byte token, so it vanishes
    let s = build("llama", &["a"], &[], &[]).unwrap();
    assert_eq!(s.encode("z"), Vec::<usize>::new());
}
