use super::*;

#[test]
fn reset_restarts_the_context_and_gives_the_same_answers() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 3);
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
    let first: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    m.reset();
    assert_eq!(m.pos(), 0);
    let again: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    assert_eq!(
        first, again,
        "stale cache entries must not leak into a new context"
    );
}

#[test]
fn forward_all_matches_the_oracle_and_feed() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 4);
    let ids = [1usize, 2, 3, 4, 5, 6, 7];
    let want = w.forward_logits(&ids).unwrap();
    let m = GpuGpt2::new(&g, &w, None).unwrap();
    let e = max_rel_err(&m.forward_all(&ids).unwrap(), &want);
    eprintln!("forward_all: max rel err {e:.2e}");
    assert!(e < FORWARD);
    assert!(m
        .forward_all(&[])
        .unwrap_err()
        .to_string()
        .contains("sequence length"));
    assert!(m.forward_all(&[0; 11]).is_err());
    assert!(m.forward_all(&[17]).is_err());
}

#[test]
fn score_matches_the_oracle() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 5);
    let ids = [4usize, 8, 15, 16, 0, 1];
    let logits = w.forward_logits(&ids[..5]).unwrap();
    let want = ferrum_ref::nn::score(&logits, &ids[1..], 17);
    let got = GpuGpt2::new(&g, &w, None).unwrap().score(&ids).unwrap();
    assert_eq!(got.len(), 5);
    for (i, ((glp, ghit), (wlp, whit))) in got.iter().zip(&want).enumerate() {
        assert!(
            (glp - wlp).abs() < 1e-4 * wlp.abs().max(1.0),
            "row {i}: {glp} vs {wlp}"
        );
        assert_eq!(ghit, whit, "row {i} argmax flag");
    }
    assert!(GpuGpt2::new(&g, &w, None).unwrap().score(&[1]).is_err());
}

#[test]
fn greedy_generation_matches_the_oracle_token_for_token() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 6);
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
    let mut ids = vec![2usize];
    let mut logits = m.feed(2).unwrap();
    for _ in 0..6 {
        let next = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
        ids.push(next);
        // The oracle sees the same history and must agree on the last row's argmax.
        let o = w.forward_logits(&ids[..ids.len() - 1]).unwrap();
        let last = &o[(ids.len() - 2) * 17..];
        let want = (0..17).fold(0, |b, i| if last[i] > last[b] { i } else { b });
        assert_eq!(next, want, "history {ids:?}");
        logits = m.feed(next).unwrap();
    }
}
