use super::*;

const L: [f32; 5] = [1.0, 3.0, 3.0, 0.5, -2.0];

#[test]
fn argmax_takes_the_first_maximum_and_ignores_nan() {
    assert_eq!(argmax(&L), 1);
    assert_eq!(argmax(&[f32::NAN, 2.0, 1.0]), 1);
    assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
    assert_eq!(argmax(&[5.0]), 0);
}

#[test]
fn temperature_zero_is_greedy_whatever_the_rng() {
    let mut r = Rng::new(1);
    for _ in 0..10 {
        assert_eq!(sample(&L, &Params::default(), &[], &mut r), 1);
    }
}

#[test]
fn top_k_one_equals_greedy() {
    let p = Params {
        temperature: 1.5,
        top_k: 1,
        ..Params::default()
    };
    let mut r = Rng::new(2);
    for _ in 0..20 {
        assert_eq!(sample(&L, &p, &[], &mut r), 1);
    }
}

#[test]
fn seeded_sampling_is_reproducible_and_covers_the_support() {
    let p = Params {
        temperature: 1.0,
        ..Params::default()
    };
    let run = |seed| {
        let mut r = Rng::new(seed);
        (0..200)
            .map(|_| sample(&L, &p, &[], &mut r))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(9), run(9));
    let draws = run(9);
    assert!(
        draws.contains(&1) && draws.contains(&2),
        "both tied maxima are drawn"
    );
    assert!(draws.iter().all(|&t| t < 5));
    let top = draws.iter().filter(|&&t| t == 1 || t == 2).count();
    assert!(
        top > 120,
        "most mass sits on the top two logits, got {top}/200"
    );
}

#[test]
fn top_p_cuts_the_tail() {
    let p = Params {
        temperature: 1.0,
        top_p: 0.5,
        ..Params::default()
    };
    let mut r = Rng::new(3);
    for _ in 0..200 {
        let t = sample(&L, &p, &[], &mut r);
        assert!(t == 1 || t == 2, "token {t} is outside the nucleus");
    }
}

#[test]
fn repetition_penalty_demotes_recent_tokens() {
    let p = Params {
        repetition_penalty: 4.0,
        ..Params::default()
    };
    let mut r = Rng::new(4);
    assert_eq!(sample(&L, &p, &[1], &mut r), 2); // 3.0 / 4 < 3.0, the tied twin wins
                                                 // Negative logits are pushed further down, not up.
    let neg = [-1.0, -1.5];
    assert_eq!(sample(&neg, &p, &[0], &mut r), 1);
    // Out-of-range recent tokens are ignored.
    assert_eq!(sample(&L, &p, &[99], &mut r), 1);
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let mut r = Rng::new(5);
    let hot = Params {
        temperature: 1.0,
        ..Params::default()
    };
    assert_eq!(sample(&[f32::NAN, f32::NAN], &hot, &[], &mut r), 0);
    assert_eq!(sample(&[0.0], &hot, &[], &mut r), 0);
    let big = Params {
        temperature: 1.0,
        top_k: 100,
        top_p: 0.0001,
        ..Params::default()
    };
    assert_eq!(sample(&L, &big, &[], &mut r), 1);
}
