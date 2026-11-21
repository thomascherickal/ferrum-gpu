use super::*;

#[test]
fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 21);
    let (batch, t) = (2, 4);
    let (ids, targets) = data(&cfg, batch, t);
    let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let h = 1e-2;
    let eval = |m: &Gpt2| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
    let n_params = m.params().len();
    let mut checked = 0;
    for pi in 0..n_params {
        let len = m.params()[pi].len();
        // Check up to 6 spread-out elements of every tensor (all of them when small).
        let picks: Vec<usize> = if len <= 6 {
            (0..len).collect()
        } else {
            (0..6).map(|j| j * (len - 1) / 5).collect()
        };
        for &i in &picks {
            let (mut p, mut q) = (m.clone(), m.clone());
            p.params_mut()[pi][i] += h;
            q.params_mut()[pi][i] -= h;
            let num = (eval(&p) - eval(&q)) / (2.0 * h);
            let ana = g.params()[pi][i];
            assert!(
                (ana - num).abs() < 4e-3 + 2e-2 * num.abs(),
                "param {pi}[{i}]: analytic {ana} vs numeric {num}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} elements checked");
    let norm: f32 = g
        .params()
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    assert!(norm > 1e-3, "gradient is suspiciously zero");
}

#[test]
fn token_embedding_gradient_sums_repeated_tokens() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 22);
    // The same token three times: its row is the sum of three position contributions.
    let ids = vec![2usize, 2, 2, 5];
    let targets = vec![1usize, 3, 4, 0];
    let (_, g) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    assert!(g.tok[2 * 4..3 * 4].iter().any(|v| v.abs() > 1e-6));
    assert!(
        g.tok[..4].iter().all(|&v| v == 0.0),
        "unused token has no gradient"
    );
}

#[test]
fn batching_equals_two_separate_sequences() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 23);
    let (ids, targets) = data(&cfg, 2, 3);
    let (l, g) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l1, g1) = loss_and_grads(&m, &ids[..3], &targets[..3], 1, 1).unwrap();
    let (l2, g2) = loss_and_grads(&m, &ids[3..], &targets[3..], 1, 1).unwrap();
    assert!(
        (l - (l1 + l2) / 2.0).abs() < 1e-5,
        "mean loss is the average of the sequences"
    );
    for ((a, b), c) in g.params().iter().zip(g1.params()).zip(g2.params()) {
        for i in 0..a.len() {
            assert!(
                (a[i] - (b[i] + c[i]) / 2.0).abs() < 1e-5,
                "gradient is the average too"
            );
        }
    }
}

#[test]
fn thread_count_does_not_change_the_result() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 24);
    let (ids, targets) = data(&cfg, 2, 4);
    let one = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    for t in [2, 5] {
        let other = loss_and_grads(&m, &ids, &targets, 2, t).unwrap();
        assert_eq!(one.0, other.0);
        assert_eq!(one.1, other.1, "bit-identical at {t} threads");
    }
}

#[test]
fn input_validation() {
    let m = Gpt2::random(tiny(), 25);
    let e = |ids: &[usize], tg: &[usize], b: usize| loss_and_grads(&m, ids, tg, b, 1).unwrap_err();
    assert!(e(&[], &[], 1).contains("split"));
    assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
    assert!(e(&[1; 6], &[1; 6], 1).contains("exceeds the context"));
    assert!(e(&[1, 2], &[1], 1).contains("targets"));
    assert!(e(&[7], &[1], 1).contains("out of range"));
    assert!(e(&[1], &[9], 1).contains("out of range"));
}

#[test]
fn loss_matches_the_forward_oracle() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 26);
    let (ids, targets) = data(&cfg, 1, 5);
    let logits = m.forward_logits(&ids).unwrap();
    let (want, _) = nn::cross_entropy(&logits, &targets, cfg.vocab);
    let (got, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    assert!((got - want).abs() < 1e-5);
}
