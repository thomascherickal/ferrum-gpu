use super::*;

#[test]
fn snapshot_and_restore_resume_a_run_exactly() {
    let Some(g) = open_for_test() else { return };
    let m = Llama::random(cfg(RopeType::Neox), 4, true, false);
    let (ids, targets) = data(&m.cfg, 2, 5);
    let mut a = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    for _ in 0..3 {
        a.step(&ids, &targets, 2, &step_cfg(0.02)).unwrap();
    }
    let snap = a.snapshot().unwrap();
    assert_eq!(snap.steps, 3);
    assert!(
        snap.m.params().iter().any(|p| p.iter().any(|&v| v != 0.0)),
        "moments are populated"
    );
    let straight: Vec<f32> = (0..2)
        .map(|_| a.step(&ids, &targets, 2, &step_cfg(0.02)).unwrap().loss)
        .collect();
    let mut b = GpuLlamaTrainer::new(&g, &m, None).unwrap();
    b.restore(&snap).unwrap();
    assert_eq!(b.steps(), 3);
    let resumed: Vec<f32> = (0..2)
        .map(|_| b.step(&ids, &targets, 2, &step_cfg(0.02)).unwrap().loss)
        .collect();
    assert_eq!(
        straight, resumed,
        "a resumed run is bit-identical on one adapter"
    );
    assert_eq!(a.snapshot().unwrap(), b.snapshot().unwrap());
    // restoring onto a device that has been lost reports the failure
    let Some(g2) = open_for_test() else { return };
    let mut dead = GpuLlamaTrainer::new(&g2, &m, None).unwrap();
    g2.destroy();
    crate::device::wait_for_loss(&g2);
    let e = dead.restore(&snap).unwrap_err();
    assert!(e.to_string().contains("lost"), "{e}");
    // a snapshot of a different shape is refused
    let other = Llama::random(cfg(RopeType::Norm), 4, false, false);
    let bad = GpuLlamaTrainer::new(&g, &other, None)
        .unwrap()
        .snapshot()
        .unwrap();
    assert!(b.restore(&bad).unwrap_err().to_string().contains("shape"));
}

#[test]
fn param_sizes_match_the_real_tensors_and_the_plan_scales_with_the_model() {
    for (name, m) in variants() {
        let biases = !m.blocks[0].bq.is_empty();
        let real: Vec<usize> = m.params().iter().map(|p| p.len()).collect();
        assert_eq!(param_sizes(&m.cfg, biases), real, "{name}");
        assert_eq!(
            static_plan_for(&real).total(),
            static_plan(&m).total(),
            "{name}"
        );
    }
    let c = cfg(RopeType::Norm);
    let small = static_plan_for(&param_sizes(&c, false)).total();
    let big = static_plan_for(&param_sizes(&Config { layers: 4, ..c }, false)).total();
    assert!(big > small);
    // activations grow with the batch and the sequence
    assert!(activation_plan(&c, 2, 4).total() > activation_plan(&c, 1, 4).total());
    assert!(activation_plan(&c, 1, 8).total() > activation_plan(&c, 1, 4).total());
}
