use super::*;

#[test]
fn bad_input_and_tied_models_are_refused() {
    let Some(g) = open_for_test() else { return };
    let m = Llama::random(cfg(RopeType::Norm), 6, false, false);
    let mut tr = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    let (ids, targets) = data(&m.cfg, 1, 4);
    let e = |r: Result<f32, GpuError>| r.expect_err("must refuse").to_string();
    assert!(e(tr.loss_and_grads(&ids, &targets, 3)).contains("split"));
    assert!(e(tr.loss_and_grads(&[0; 9], &[0; 9], 1)).contains("context"));
    assert!(e(tr.loss_and_grads(&ids, &targets[..3], 1)).contains("targets"));
    assert!(e(tr.loss_and_grads(&[11, 0, 0, 0], &targets, 1)).contains("out of range"));
    let tied = Llama::random(cfg(RopeType::Norm), 7, false, true);
    let r = GpuLlamaTrainer::new(&g, &tied, None);
    assert!(r.err().expect("refused").to_string().contains("tied"));
    let mut bad = m.clone();
    bad.rope.linear_factor = 0.0;
    assert!(GpuLlamaTrainer::new(&g, &bad, None).is_err());
    assert_eq!(tr.config(), &m.cfg);
}

#[test]
fn the_footprint_is_checked_before_anything_is_allocated() {
    let Some(g) = open_for_test() else { return };
    let m = Llama::random(cfg(RopeType::Norm), 8, true, false);
    let before = g.buffers_created();
    let tiny = Budget {
        bytes: 1000,
        source: Source::Override,
    };
    let e = GpuLlamaTrainer::new(&g, &m, Some(tiny))
        .err()
        .expect("must refuse");
    assert!(e.to_string().contains("budget is 1000"), "{e}");
    assert_eq!(g.buffers_created(), before);
    // the static part fits, but one long batch does not: the step is refused, nothing is allocated
    let stat = static_plan(&m).total();
    let (ids, targets) = data(&m.cfg, 4, 8);
    let tight = Budget {
        bytes: stat + activation_plan(&m.cfg, 1, 1).total(),
        source: Source::Override,
    };
    let mut tr = GpuLlamaTrainer::new(&g, &m, Some(tight)).unwrap();
    let e = tr
        .step(&ids, &targets, 4, &step_cfg(0.01))
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not fit"), "{e}");
    assert_eq!(tr.steps(), 0);
    // a shape that fits still works afterwards
    let (i1, t1) = data(&m.cfg, 1, 1);
    tr.step(&i1, &t1, 1, &step_cfg(0.01)).unwrap();
    assert_eq!(tr.steps(), 1);
}

#[test]
fn a_fault_at_any_command_leaves_the_training_state_untouched() {
    let Some(g) = open_for_test() else { return };
    let m = Llama::random(cfg(RopeType::Neox), 9, true, false);
    let (ids, targets) = data(&m.cfg, 2, 3);
    let mut tr = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    tr.step(&ids, &targets, 2, &step_cfg(0.01)).unwrap(); // populate the moments first
    let commands = g.sweep_faults(|| {
        let before = tr.snapshot().unwrap();
        let r = tr.step(&ids, &targets, 2, &step_cfg(0.01));
        if r.is_err() {
            assert_eq!(
                tr.snapshot().unwrap(),
                before,
                "a failed step must change nothing"
            );
        }
        r.map(|_| ())
    });
    assert!(commands > 100, "{commands}");
    assert_eq!(tr.steps(), 2, "only the successful attempt counted");
}

#[test]
fn restoring_a_model_without_biases_skips_the_absent_tensors() {
    let Some(g) = open_for_test() else { return };
    let m = Llama::random(cfg(RopeType::Norm), 12, false, false);
    let (ids, targets) = data(&m.cfg, 1, 4);
    let mut a = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    a.step(&ids, &targets, 1, &step_cfg(0.02)).unwrap();
    let snap = a.snapshot().unwrap();
    assert!(snap.weights.blocks[0].bq.is_empty());
    let mut b = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    b.restore(&snap).unwrap();
    assert_eq!(b.snapshot().unwrap(), snap);
}
