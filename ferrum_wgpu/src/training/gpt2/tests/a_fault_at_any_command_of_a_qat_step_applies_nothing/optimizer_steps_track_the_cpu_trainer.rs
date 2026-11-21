use super::*;

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
