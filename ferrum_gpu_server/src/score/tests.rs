use super::*;
use crate::decoder::{GenReq, Info};

/// Bytes as tokens, context `ctx`, and a score that depends only on the target token, so the
/// expected sums are easy to write down: log-prob `-(1 + t % 3)`, argmax when `t` is even.
struct Table {
    ctx: usize,
    calls: std::sync::Mutex<Vec<usize>>,
}

impl Table {
    fn new(ctx: usize) -> Self {
        Table {
            ctx,
            calls: Default::default(),
        }
    }
}

impl Decoder for Table {
    fn info(&self) -> Info {
        Info {
            device: "t".into(),
            kind: "cpu".into(),
            arch: "t".into(),
            params: 0,
            vocab: 256,
            ctx: self.ctx,
        }
    }
    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        Ok(text.bytes().map(usize::from).collect())
    }
    fn bos(&self) -> usize {
        0
    }
    fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
        Err("unused".into())
    }
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        assert!(
            ids.len() <= self.ctx + 1,
            "window of {} for context {}",
            ids.len(),
            self.ctx
        );
        self.calls.lock().unwrap().push(ids.len());
        Ok(ids[1..]
            .iter()
            .map(|&t| (-(1.0 + (t % 3) as f32), t % 2 == 0))
            .collect())
    }
}

fn lp(t: usize) -> f64 {
    -(1.0 + (t % 3) as f64)
}

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

#[test]
fn a_decoder_error_is_passed_up() {
    struct Failing;
    impl Decoder for Failing {
        fn info(&self) -> Info {
            Info {
                device: "f".into(),
                kind: "cpu".into(),
                arch: "f".into(),
                params: 0,
                vocab: 4,
                ctx: 4,
            }
        }
        fn encode(&self, _: &str) -> Result<Vec<usize>, String> {
            Err("no tokenizer".into())
        }
        fn bos(&self) -> usize {
            0
        }
        fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
            Err("x".into())
        }
        fn score(&self, _: &[usize]) -> Result<Vec<(f32, bool)>, String> {
            Err("device lost".into())
        }
    }
    assert_eq!(pair(&Failing, "a", "b").unwrap_err(), "no tokenizer");
    assert_eq!(rolling(&Failing, "a").unwrap_err(), "no tokenizer");
    assert_eq!(
        token_logprobs(&Failing, &[1, 2, 3], 1).unwrap_err(),
        "device lost"
    );
}
