use super::*;

#[test]
fn pair_scores_only_the_continuation() {
    let d = Table::new(32);
    let s = pair(&d, "ab", "cd").unwrap();
    assert_eq!(s.n_tokens, 2);
    assert_eq!(s.logprob, lp(99) + lp(100));
    assert!(!s.greedy, "99 is odd");
    assert!(pair(&d, "ab", "d").unwrap().greedy, "100 is even");
}

#[test]
fn trailing_context_whitespace_moves_to_the_continuation() {
    let d = Table::new(32);
    let moved = pair(&d, "ab ", "c").unwrap();
    let same = pair(&d, "ab", " c").unwrap();
    assert_eq!(moved, same);
    assert_eq!(moved.n_tokens, 2);
}

#[test]
fn an_empty_context_is_scored_after_bos_and_empty_continuation_is_zero() {
    let d = Table::new(32);
    let s = pair(&d, "", "ab").unwrap();
    assert_eq!((s.n_tokens, s.logprob), (2, lp(97) + lp(98)));
    let z = pair(&d, "abc", "").unwrap();
    assert_eq!((z.n_tokens, z.logprob, z.greedy), (0, 0.0, true));
}

#[test]
fn rolling_scores_every_token_after_bos() {
    let d = Table::new(32);
    let s = rolling(&d, "abc").unwrap();
    assert_eq!(s.n_tokens, 3);
    assert_eq!(s.logprob, lp(97) + lp(98) + lp(99));
    let e = rolling(&d, "").unwrap();
    assert_eq!((e.n_tokens, e.logprob), (0, 0.0));
}

#[test]
fn long_texts_are_scored_in_windows_and_every_token_exactly_once() {
    let text: String = (0..50).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
    let small = Table::new(8);
    let s = rolling(&small, &text).unwrap();
    let big = Table::new(100);
    let whole = rolling(&big, &text).unwrap();
    assert_eq!(s.n_tokens, 50);
    assert_eq!(s.n_tokens, whole.n_tokens);
    assert!(
        (s.logprob - whole.logprob).abs() < 1e-9,
        "{} vs {}",
        s.logprob,
        whole.logprob
    );
    assert_eq!(s.greedy, whole.greedy);
    let calls = small.calls.lock().unwrap();
    assert!(
        calls.len() > 1 && calls.iter().all(|&n| n <= 9),
        "{calls:?}"
    );
    assert_eq!(big.calls.lock().unwrap().len(), 1);
}

#[test]
fn token_logprobs_handles_trivial_inputs() {
    let d = Table::new(8);
    assert_eq!(token_logprobs(&d, &[1], 1).unwrap(), ZERO);
    assert_eq!(token_logprobs(&d, &[1, 2], 5).unwrap(), ZERO);
    assert_eq!(
        token_logprobs(&d, &[1, 2], 0).unwrap().n_tokens,
        1,
        "from is clamped to 1"
    );
}
