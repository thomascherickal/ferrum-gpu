use super::*;

#[test]
fn micro_batches_give_the_same_update_as_the_whole_batch() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w0 = Gpt2::init(c, 51);
    let sc = StepConfig {
        adam: Adam {
            weight_decay: 0.01,
            ..Adam::new(0.01)
        },
        clip: 0.3,
    };
    let steps: Vec<_> = (0..3).map(|s| batch_data(&c, 6, 5, s)).collect();
    let mut whole = GpuTrainer::new(&g, &w0, None).unwrap();
    let whole_stats: Vec<StepStats> = steps
        .iter()
        .map(|(i, t)| whole.step(i, t, 6, &sc).unwrap())
        .collect();
    assert_eq!(whole.last_micro(), 6, "a batch that fits is not split");
    for micro in [1usize, 2, 3] {
        let mut split = GpuTrainer::new(&g, &w0, None).unwrap();
        for (s, (ids, tg)) in steps.iter().enumerate() {
            let st = split.step_with_micro(ids, tg, 6, micro, &sc).unwrap();
            let want = &whole_stats[s];
            assert!(
                (st.loss - want.loss).abs() < 1e-5 * want.loss.max(1.0),
                "micro {micro} step {s}: loss {} vs {}",
                st.loss,
                want.loss
            );
            assert!(
                (st.grad_norm - want.grad_norm).abs() < 1e-4 * want.grad_norm.max(1.0),
                "micro {micro} step {s}: norm"
            );
        }
        assert_eq!(split.last_micro(), micro);
        let e = max_err_models(
            &split.weights_host().unwrap(),
            &whole.weights_host().unwrap(),
        );
        eprintln!("micro {micro}: weights after 3 steps differ by {e:.2e}");
        assert!(e < crate::tolerance::BACKWARD, "micro {micro}: {e}");
    }
}

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

#[test]
fn explicit_micro_batch_sizes_are_validated() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 54), None).unwrap();
    let (ids, targets) = batch_data(&c, 4, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    for bad in [0usize, 3, 5] {
        let e = tr.step_with_micro(&ids, &targets, 4, bad, &sc).unwrap_err();
        assert!(
            e.to_string().contains("must be positive and divide"),
            "{bad}: {e}"
        );
    }
    assert_eq!(tr.steps(), 0);
    tr.step_with_micro(&ids, &targets, 4, 4, &sc).unwrap();
    assert_eq!(
        tr.last_micro(),
        4,
        "micro equal to the batch is the unsplit path"
    );
}
