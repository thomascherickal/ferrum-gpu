use super::*;

#[test]
fn loss_and_every_gradient_match_the_cpu_oracle() {
    let Some(g) = open_for_test() else { return };
    for (name, m) in variants() {
        for (batch, t) in [(1usize, 1usize), (2, 4), (1, 8)] {
            let (ids, targets) = data(&m.cfg, batch, t);
            let (want_loss, want) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
            let mut tr = GpuLlamaTrainer::new(&g, &m, None).unwrap();
            let loss = tr.loss_and_grads(&ids, &targets, batch).unwrap();
            assert!(
                (loss - want_loss).abs() < 1e-5 * want_loss.max(1.0),
                "{name} {batch}x{t}: loss {loss} vs {want_loss}"
            );
            let got = tr.grads_host().unwrap();
            let mut worst = 0.0f32;
            for (pi, (a, b)) in got.params().iter().zip(want.params()).enumerate() {
                assert_eq!(a.len(), b.len(), "{name} param {pi} shape");
                let e = max_rel_err(a, b);
                worst = worst.max(e);
                assert!(
                    e < BACKWARD,
                    "{name} {batch}x{t} param {pi}: max rel err {e}"
                );
            }
            eprintln!("{name} {batch}x{t}: worst gradient error {worst:.2e}");
        }
    }
}

#[test]
fn a_training_run_follows_the_cpu_oracle() {
    let Some(g) = open_for_test() else { return };
    for (name, m) in variants() {
        let (ids, targets) = data(&m.cfg, 2, 6);
        let mut cpu = LlamaTrainer::new(m.clone(), 1);
        let mut gpu = GpuLlamaTrainer::new(&g, &m, None).unwrap();
        let (mut first, mut last, mut worst) = (0.0f32, 0.0f32, 0.0f32);
        for s in 0..12 {
            let cfg = step_cfg(0.02);
            let (a, b) = (
                cpu.step(&ids, &targets, 2, &cfg).unwrap(),
                gpu.step(&ids, &targets, 2, &cfg).unwrap(),
            );
            worst = worst.max((a.loss - b.loss).abs() / a.loss.max(1.0));
            assert!(
                (a.grad_norm - b.grad_norm).abs() < 1e-3 * a.grad_norm.max(1.0),
                "{name} step {s}: norm {} vs {}",
                a.grad_norm,
                b.grad_norm
            );
            if s == 0 {
                first = b.loss;
            }
            last = b.loss;
        }
        eprintln!(
            "{name}: loss {first:.4} -> {last:.4}, worst cpu/gpu loss difference {worst:.2e}"
        );
        assert!(worst < 1e-4, "{name}: loss curves diverge by {worst}");
        assert!(
            last < 0.8 * first,
            "{name}: the loss must fall ({first} -> {last})"
        );
        assert_eq!(gpu.steps(), 12);
        // function space: the trained models agree on every logit (parameters whose true gradient
        // is zero, such as the key biases, legitimately drift apart under Adam)
        let probe = [1usize, 4, 2, 7, 3];
        let (wa, wb) = (
            cpu.weights().forward_logits(&probe).unwrap(),
            gpu.weights_host().unwrap().forward_logits(&probe).unwrap(),
        );
        let e = max_rel_err(&wb, &wa);
        eprintln!("{name}: trained-model logits differ by {e:.2e}");
        assert!(e < 1e-3, "{name}: {e}");
    }
}

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
