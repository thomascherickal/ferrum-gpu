use super::*;

#[test]
fn a_fault_at_any_command_of_a_qat_step_applies_nothing() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 65), None).unwrap();
    tr.set_qat(true).unwrap();
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let commands = g.sweep_faults(|| {
        let before = tr.steps();
        let r = tr.step(&ids, &targets, 1, &sc).map(|_| ());
        if r.is_err() {
            assert_eq!(tr.steps(), before);
        }
        r
    });
    // The snaps add one command per parameter (2 + 16 + 3 = 21 for one layer) to a plain step.
    let mut plain = GpuTrainer::new(&g, &Gpt2::init(c, 65), None).unwrap();
    let plain_commands = g.sweep_faults(|| plain.step(&ids, &targets, 1, &sc).map(|_| ()));
    assert_eq!(commands, plain_commands + 21);
    // loss_and_grads snaps too.
    let n = g.sweep_faults(|| tr.loss_and_grads(&ids, &targets, 1).map(|_| ()));
    assert!(n > 21);
}

#[test]
fn weights_round_trip_through_the_device() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 2);
    let tr = GpuTrainer::new(&g, &w, None).unwrap();
    assert_eq!(tr.weights_host().unwrap(), w);
    assert_eq!(tr.config(), &w.cfg);
    assert_eq!(tr.steps(), 0);
    let (m, v) = tr.moments_host().unwrap();
    assert_eq!(m, w.zeros_like());
    assert_eq!(v, w.zeros_like());
}

#[test]
fn loss_and_every_gradient_match_the_cpu_backward() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::random(c, 3);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    for (batch, t) in [(1usize, 1usize), (1, 3), (2, 5), (3, 8)] {
        let (ids, targets) = batch_data(&c, batch, t, batch + t);
        let (want_loss, want) = loss_and_grads(&w, &ids, &targets, batch, 1).unwrap();
        let loss = tr.loss_and_grads(&ids, &targets, batch).unwrap();
        assert!(
            (loss - want_loss).abs() < 1e-5 * want_loss.max(1.0),
            "{batch}x{t}: loss {loss} vs {want_loss}"
        );
        let got = tr.grads_host().unwrap();
        let mut worst = 0.0f32;
        for (i, (gp, wp)) in got.params().iter().zip(want.params()).enumerate() {
            let e = max_rel_err(gp, wp);
            worst = worst.max(e);
            assert!(
                e < crate::tolerance::BACKWARD,
                "{batch}x{t} param {i}: max rel err {e}"
            );
        }
        eprintln!("grads {batch}x{t}: worst max rel err {worst:.2e}");
    }
}

#[test]
fn optimizer_steps_track_the_cpu_trainer() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w0 = Gpt2::init(c, 4);
    let adam = Adam {
        weight_decay: 0.01,
        ..Adam::new(0.01)
    };
    let mut cpu = w0.clone();
    let mut st = AdamState::new(&w0);
    let mut tr = GpuTrainer::new(&g, &w0, None).unwrap();
    for s in 0..5usize {
        let (ids, targets) = batch_data(&c, 2, 6, s);
        let (cpu_loss, grads) = loss_and_grads(&cpu, &ids, &targets, 2, 1).unwrap();
        let norm = global_norm(&grads);
        adam_step(&mut cpu, &grads, &mut st, &adam, clip_scale(norm, 0.05));
        let stats = tr
            .step(&ids, &targets, 2, &StepConfig { adam, clip: 0.05 })
            .unwrap();
        assert!(
            (stats.loss - cpu_loss).abs() < 1e-4 * cpu_loss.max(1.0),
            "step {s}: loss {} vs {cpu_loss}",
            stats.loss
        );
        assert!(
            (stats.grad_norm - norm).abs() < 1e-3 * norm.max(1.0),
            "step {s}: norm {} vs {norm}",
            stats.grad_norm
        );
    }
    assert_eq!(tr.steps(), 5);
    let got = tr.weights_host().unwrap();
    let mut worst = 0.0f32;
    for (i, (gp, wp)) in got.params().iter().zip(cpu.params()).enumerate() {
        let e = max_rel_err(gp, wp);
        worst = worst.max(e);
        assert!(
            e < crate::tolerance::BACKWARD,
            "weights after 5 steps, param {i}: max rel err {e}"
        );
    }
    eprintln!("weights after 5 clipped AdamW steps: worst max rel err {worst:.2e}");
}

#[test]
fn training_a_tiny_model_on_the_gpu_reduces_the_loss() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let (ids, targets) = batch_data(&c, 2, 6, 0);
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 5), None).unwrap();
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let first = tr.step(&ids, &targets, 2, &sc).unwrap().loss;
    let mut last = first;
    for _ in 0..60 {
        last = tr.step(&ids, &targets, 2, &sc).unwrap().loss;
    }
    assert!(first > 2.0, "starts near ln(13) = 2.56, got {first}");
    assert!(last < first * 0.4, "memorises one batch: {first} -> {last}");
}

#[test]
fn two_identical_runs_end_with_identical_bytes() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let run = || {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 6), None).unwrap();
        for s in 0..4usize {
            let (ids, targets) = batch_data(&c, 2, 5, s);
            tr.step(
                &ids,
                &targets,
                2,
                &StepConfig {
                    adam: Adam::new(0.01),
                    clip: 0.5,
                },
            )
            .unwrap();
        }
        (tr.weights_host().unwrap(), tr.moments_host().unwrap())
    };
    assert_eq!(run(), run());
}
