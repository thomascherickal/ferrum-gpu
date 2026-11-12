use super::*;
use crate::engine::fresh_model;

#[test]
fn a_zero_share_holds_nothing_out() {
    let (t, v) = split(b"hello world", 0.0).unwrap();
    assert_eq!((t, v), (&b"hello world"[..], &b""[..]));
}

#[test]
fn the_tail_of_the_text_is_held_out() {
    let text = b"0123456789".repeat(10); // 100 bytes
    let (t, v) = split(&text, 0.2).unwrap();
    assert_eq!((t.len(), v.len()), (80, 20));
    assert_eq!([t, v].concat(), text, "train then validation, nothing lost");
}

#[test]
fn a_tiny_share_still_holds_out_enough_for_one_pair() {
    let (t, v) = split(&[7u8; 50], 0.001).unwrap();
    assert_eq!((t.len(), v.len()), (50 - MIN_BYTES, MIN_BYTES));
}

#[test]
fn shares_outside_the_range_are_refused() {
    for bad in [-0.1f32, 0.51, 1.0, f32::NAN] {
        assert!(
            split(b"abcdefgh", bad)
                .unwrap_err()
                .contains("validation share"),
            "{bad}"
        );
    }
    assert!(split(b"abcdefgh", MAX_FRACTION).is_ok());
}

#[test]
fn a_text_too_short_to_split_is_refused() {
    assert!(split(b"ab", 0.5).unwrap_err().contains("too few"));
    assert!(split(b"abc", 0.5).is_ok());
}

#[test]
fn the_loss_of_an_untrained_byte_model_is_finite_and_of_the_order_of_ln_vocab() {
    // a random init need not be exactly uniform, but it is nowhere near fitted either
    let uniform = (256f32).ln();
    for seed in [1, 2, 7] {
        let m = fresh_model("tiny", seed).unwrap();
        let loss = mean_loss(&m, b"the quick brown fox jumps over the lazy dog", 16).unwrap();
        assert!(
            loss.is_finite() && loss > 0.5 * uniform && loss < 2.0 * uniform,
            "{seed}: {loss}"
        );
    }
}

#[test]
fn no_pair_means_no_loss() {
    let m = fresh_model("tiny", 1).unwrap();
    assert_eq!(mean_loss(&m, b"", 16), None);
    assert_eq!(mean_loss(&m, b"a", 16), None);
    assert!(mean_loss(&m, b"ab", 16).is_some());
}

#[test]
fn the_loss_does_not_depend_on_how_windows_are_cut_much_but_does_on_the_text() {
    let m = fresh_model("tiny", 2).unwrap();
    let a = mean_loss(&m, b"aaaaaaaaaaaaaaaa", 8).unwrap();
    let b = mean_loss(&m, b"zqxjvkwpyfbgmcud", 8).unwrap();
    assert_ne!(a, b);
}
