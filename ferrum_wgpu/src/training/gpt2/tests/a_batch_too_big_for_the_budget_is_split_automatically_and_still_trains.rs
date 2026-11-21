use super::*;

#[test]
fn a_batch_too_big_for_the_budget_is_split_automatically_and_still_trains() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w0 = Gpt2::init(c, 52);
    let (batch, t) = (4usize, 6usize);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    // A budget that fits one sequence at a time (plus accumulators) but not four together.
    let need = |m: usize| {
        let mut p = static_plan(&c);
        accum_plan(&c).rows().iter().for_each(|(n, b)| p.add(n, *b));
        activation_plan(&c, m, t)
            .rows()
            .iter()
            .for_each(|(n, b)| p.add(n, *b));
        p.total()
    };
    let budget = Some(Budget {
        bytes: need(1) + 64,
        source: Source::Override,
    });
    assert!(need(4) > need(1) + 64, "the test must really force a split");
    let (ids, targets) = batch_data(&c, batch, t, 0);
    let mut tight = GpuTrainer::new(&g, &w0, budget).unwrap();
    let mut roomy = GpuTrainer::new(&g, &w0, None).unwrap();
    let a = tight.step(&ids, &targets, batch, &sc).unwrap();
    let b = roomy.step(&ids, &targets, batch, &sc).unwrap();
    assert_eq!(tight.last_micro(), 1, "split down to one sequence");
    assert_eq!(roomy.last_micro(), batch);
    assert!((a.loss - b.loss).abs() < 1e-5 * b.loss.max(1.0));
    assert!(
        max_err_models(
            &tight.weights_host().unwrap(),
            &roomy.weights_host().unwrap()
        ) < 1e-4
    );
    // And it keeps training: the loss on a repeated batch falls.
    let first = a.loss;
    let mut last = first;
    for _ in 0..40 {
        last = tight.step(&ids, &targets, batch, &sc).unwrap().loss;
    }
    assert!(last < first * 0.5, "{first} -> {last}");
}

#[test]
fn a_batch_that_does_not_fit_even_one_sequence_at_a_time_is_refused() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 53);
    let mut tr = GpuTrainer::new(
        &g,
        &w,
        Some(Budget {
            bytes: static_plan(&c).total() + 16,
            source: Source::Override,
        }),
    )
    .unwrap();
    let (ids, targets) = batch_data(&c, 2, 4, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let e = tr.step(&ids, &targets, 2, &sc).unwrap_err().to_string();
    assert!(e.contains("does not fit") && e.contains("budget"), "{e}");
    assert_eq!(tr.steps(), 0);
}
