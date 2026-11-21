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
