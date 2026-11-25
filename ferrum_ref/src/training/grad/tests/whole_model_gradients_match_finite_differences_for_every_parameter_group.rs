use super::*;

#[test]
fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 21);
    let (batch, t) = (2, 4);
    let (ids, targets) = data(&cfg, batch, t);
    let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let eval = |m: &Gpt2| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
    let checked: usize = (0..m.params().len())
        .map(|pi| crate::grad::check_tensor_fd(&m, pi, g.params()[pi], eval, "gpt2"))
        .sum();
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
