use super::*;

#[test]
fn the_loss_curve_matches_the_cpu_trainer_step_for_step() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let toks = pattern(500, c.vocab);
    let cfg = loop_cfg();
    let mut cpu = CpuTrainer::new(Gpt2::init(c, 21), 1);
    let want = train(&mut cpu, &toks, &cfg, None, None, &mut NoHooks).unwrap();
    let mut gpu = GpuTrainer::new(&g, &Gpt2::init(c, 21), None).unwrap();
    let got = train(&mut gpu, &toks, &cfg, None, None, &mut NoHooks).unwrap();
    assert_eq!(got.steps_run, want.steps_run);
    let mut worst = 0.0f32;
    for (i, (a, b)) in got.losses.iter().zip(&want.losses).enumerate() {
        let e = (a - b).abs() / b.abs().max(1.0);
        worst = worst.max(e);
        assert!(e < 1e-4, "step {}: gpu loss {a} vs cpu loss {b}", i + 1);
    }
    eprintln!("60-step loss curve: worst relative difference {worst:.2e}");
    let (first, last) = (got.losses[0], *got.losses.last().unwrap());
    assert!(
        last < first * 0.3,
        "both learn the pattern: {first} -> {last}"
    );
    let (gw, cw) = (gpu.weights_host().unwrap(), cpu.weights().clone());
    for (i, (a, b)) in gw.params().iter().zip(cw.params()).enumerate() {
        // The key bias (canonical offset 5 in each block) has an exactly-zero true gradient:
        // softmax ignores a constant added to every score. Only rounding noise reaches Adam,
        // which rescales any tiny gradient to a full-size step, so those values legitimately
        // differ between backends without changing the model's function.
        let is_key_bias = i >= 2 && i < 2 + c.layers * 16 && (i - 2) % 16 == 5;
        if is_key_bias {
            continue;
        }
        assert!(
            max_rel_err(a, b) < 1e-4,
            "weights after 60 steps, param {i}"
        );
    }
    // Function-space parity: the trained models agree on a held-out sequence.
    let probe: Vec<usize> = (0..8).map(|i| (i * 5 + 2) % c.vocab).collect();
    let (lg, lc) = (
        gw.forward_logits(&probe).unwrap(),
        cw.forward_logits(&probe).unwrap(),
    );
    let e = max_rel_err(&lg, &lc);
    eprintln!("trained-model logits: worst relative difference {e:.2e}");
    assert!(e < 1e-4, "trained models disagree on logits: {e}");
}

#[test]
fn a_gpu_run_resumed_from_a_checkpoint_equals_the_uninterrupted_run() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let toks = pattern(400, c.vocab);
    let cfg = TrainConfig {
        max_steps: 20,
        ..loop_cfg()
    };
    let mut straight = GpuTrainer::new(&g, &Gpt2::init(c, 22), None).unwrap();
    train(&mut straight, &toks, &cfg, None, None, &mut NoHooks).unwrap();
    // Stop at step 7 (mid-epoch), keep the checkpoint, finish in a brand-new trainer.
    let mut a = GpuTrainer::new(&g, &Gpt2::init(c, 22), None).unwrap();
    let mut keep = Keep(None, Some(7));
    let r = train(&mut a, &toks, &cfg, None, Some("sha256:x"), &mut keep).unwrap();
    assert!(r.stopped);
    let cp = keep.0.unwrap();
    assert_eq!(
        (cp.snapshot.steps, cp.corpus_id.as_deref()),
        (7, Some("sha256:x"))
    );
    let mut b = GpuTrainer::new(&g, &Gpt2::init(c, 999), None).unwrap();
    b.restore(&cp.snapshot).unwrap();
    let r2 = train(&mut b, &toks, &cfg, Some(&cp.shape), None, &mut NoHooks).unwrap();
    assert_eq!(r2.steps_run, 13);
    assert_eq!(b.steps(), 20);
    assert_eq!(
        b.snapshot().unwrap(),
        straight.snapshot().unwrap(),
        "weights, moments and step count"
    );
}
