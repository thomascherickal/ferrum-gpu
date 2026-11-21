use super::*;

#[test]
fn a_prefix_gives_the_same_logits_as_the_prefix_of_a_longer_run() {
    let m = Llama::random(cfg(), 4, false, false);
    let long = m.forward_logits(&[1, 2, 3, 4, 5]).unwrap();
    let short = m.forward_logits(&[1, 2, 3]).unwrap();
    assert!(max_diff(&long[..3 * 13], &short) < 1e-5, "causality");
}

#[test]
fn rope_scaling_changes_the_output_and_unit_scaling_does_not() {
    let mut m = Llama::random(cfg(), 5, false, false);
    let ids = [1, 2, 3, 4];
    let base = m.forward_logits(&ids).unwrap();
    m.rope = RopeScaling {
        freq_factors: vec![1.0, 1.0],
        linear_factor: 1.0,
    };
    assert_eq!(m.forward_logits(&ids).unwrap(), base);
    m.rope.linear_factor = 4.0;
    assert!(max_diff(&m.forward_logits(&ids).unwrap(), &base) > 1e-4);
}

#[test]
fn bad_input_is_refused_with_the_reason() {
    let m = Llama::random(cfg(), 6, false, false);
    assert!(m
        .forward_logits(&[])
        .unwrap_err()
        .contains("sequence length"));
    assert!(m
        .forward_logits(&[0; 13])
        .unwrap_err()
        .contains("sequence length"));
    assert!(m
        .forward_logits(&[13])
        .unwrap_err()
        .contains("out of range"));
    let mut cache = Cache::new(2);
    for _ in 0..12 {
        m.feed(&mut cache, 1).unwrap();
    }
    assert!(m.feed(&mut cache, 1).unwrap_err().contains("full"));
    assert!(m.feed(&mut Cache::new(2), 99).is_err());
}

#[test]
fn biases_are_applied_to_q_k_and_v() {
    let a = Llama::random(cfg(), 7, true, false);
    let mut b = a.clone();
    for blk in &mut b.blocks {
        blk.bv.iter_mut().for_each(|v| *v += 1.0);
    }
    assert!(
        max_diff(
            &a.forward_logits(&[1, 2]).unwrap(),
            &b.forward_logits(&[1, 2]).unwrap()
        ) > 1e-3
    );
}
