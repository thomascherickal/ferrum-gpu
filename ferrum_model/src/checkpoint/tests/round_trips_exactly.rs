use super::*;

#[test]
fn round_trips_exactly() {
    let cp = sample();
    assert_eq!(from_bytes(to_bytes(&cp)).unwrap(), cp);
    let mut no_id = cp.clone();
    no_id.corpus_id = None;
    assert_eq!(from_bytes(to_bytes(&no_id)).unwrap(), no_id);
}

#[test]
fn names_follow_the_p_i_data_m_v_scheme() {
    let g = Gguf::parse(to_bytes(&sample())).unwrap();
    assert_eq!(g.string("general.architecture").unwrap(), KIND);
    let n = param_lens(&cfg()).unwrap().len();
    assert_eq!(g.tensors.len(), 3 * n);
    for name in ["p0.data", "p0.m", "p0.v", &format!("p{}.v", n - 1)] {
        assert!(g.tensor(name).is_some(), "{name}");
    }
    assert_eq!(g.uint("ferrum.ckpt.step").unwrap(), 42);
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let good = to_bytes(&sample());
    // Cutting only the final alignment padding leaves a valid file; stop before real data ends.
    let cuts = [0, 1, 10, 100, 1000, good.len() / 2, good.len() - 200];
    for cut in cuts {
        assert!(from_bytes(good[..cut].to_vec()).is_err(), "cut at {cut}");
    }
}

#[test]
fn wrong_kind_and_missing_keys_are_named() {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("not a ferrum-gpu checkpoint"));
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("missing key"));
}
