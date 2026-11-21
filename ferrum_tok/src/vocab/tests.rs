use crate::test_support::{small_tok, trained_tok};
use crate::ByteBpeTokenizer;

fn parts(t: &ByteBpeTokenizer) -> (Vec<String>, Vec<String>, Vec<i32>) {
    (t.gguf_tokens(), t.gguf_merges(), t.gguf_token_types())
}

#[test]
fn arrays_round_trip_with_and_without_specials() {
    for t in [small_tok(), trained_tok(), ByteBpeTokenizer::byte_level()] {
        let (tok, mer, ty) = parts(&t);
        assert_eq!(tok.len(), t.vocab_size());
        assert_eq!(
            mer.len(),
            t.vocab_size() - 256 - (tok.len() - 256 - mer.len())
        );
        let back = ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty).unwrap();
        assert_eq!(back.encode_state(), t.encode_state());
        let text = "abab cdcd ünï low lower";
        assert_eq!(back.encode(text), t.encode(text));
    }
}

#[test]
fn specials_are_control_tokens_after_the_merges() {
    let t = trained_tok();
    let (tok, mer, ty) = parts(&t);
    assert_eq!(tok.last().unwrap(), "<|endoftext|>");
    assert_eq!(*ty.last().unwrap(), 3);
    assert!(ty[..256 + mer.len()].iter().all(|&x| x == 1));
}

#[test]
fn rejects_inconsistent_arrays() {
    let (tok, mer, ty) = parts(&small_tok());
    let e = |r: crate::error::Result<ByteBpeTokenizer>| r.expect_err("must fail").to_string();
    // types length differs from tokens
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty[1..])).contains("types"));
    // byte token wrong
    let mut bad = tok.clone();
    bad.swap(65, 66);
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&bad, &mer, &ty)).contains("is not byte"));
    // malformed merge
    let mut badm = mer.clone();
    badm[0] = "nospace".into();
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &badm, &ty)).contains("bad merge"));
    // merge part that is not a token
    let mut badp = mer.clone();
    badp[0] = "zzzz yyyy".into();
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &badp, &ty)).contains("not a token"));
    // merge that uses a later token
    let mut late = mer.clone();
    late.swap(0, 1);
    assert!(ByteBpeTokenizer::from_gguf_vocab(&tok, &late, &ty).is_err());
    // a merge whose left part is the token a later merge makes
    let mut later = mer.clone();
    later[0] = format!("{} {}", tok[257], tok[65]);
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &later, &ty)).contains("later token"));
    // token after the merges that is not control
    let (t2, m2, mut y2) = parts(&trained_tok());
    *y2.last_mut().unwrap() = 1;
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&t2, &m2, &y2)).contains("not control"));
    // duplicate special token
    let (mut t3, m3, mut y3) = parts(&trained_tok());
    t3.push("<|endoftext|>".into());
    y3.push(3);
    assert!(e(ByteBpeTokenizer::from_gguf_vocab(&t3, &m3, &y3)).contains("duplicate"));
}

#[test]
fn foreign_text_token_types_are_not_inspected() {
    let (tok, mer, mut ty) = parts(&small_tok());
    for x in ty.iter_mut().take(256 + mer.len()) {
        *x = 6; // another writer's label for text tokens; specials stay control
    }
    let r = ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty);
    assert!(r.is_ok(), "{:?}", r.err());
}
