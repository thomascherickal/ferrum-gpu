use super::*;
use crate::budget::Source;
use crate::device::open_for_test;
use crate::tolerance::max_rel_err;
use ferrum_ref::grad::loss_and_grads;
use ferrum_ref::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};
use ferrum_ref::trainer::{train, Checkpoint, CpuTrainer, Hooks, NoHooks, TrainConfig};

fn cfg() -> Config {
    Config {
        vocab: 13,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 16,
    }
}

fn batch_data(c: &Config, batch: usize, t: usize, salt: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1 + salt) % c.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2 + salt) % c.vocab).collect(),
    )
}

#[test]
fn plans_are_consistent_with_the_allocated_buffers() {
    let c = cfg();
    let sp = static_plan(&c);
    assert_eq!(sp.rows().len(), 4 * (2 + 2 * 16 + 3));
    let w = Gpt2::random(c, 1);
    assert_eq!(sp.total(), 4 * 4 * w.num_params() as u64);
    let ap = activation_plan(&c, 2, 5);
    assert!(ap.rows().iter().any(|(n, _)| n == "blk1.probs"));
    assert!(activation_plan(&c, 4, 5).total() > ap.total());
}

#[test]
fn load_refuses_moments_of_another_shape() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 31);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let other = Gpt2::init(Config { dim: 12, ..c }, 31);
    let e = tr
        .load(&w, Some((&other.zeros_like(), &w.zeros_like())), 0)
        .unwrap_err();
    assert!(e.to_string().contains("moment shapes differ"), "{e}");
    let e = tr
        .load(&w, Some((&w.zeros_like(), &other.zeros_like())), 0)
        .unwrap_err();
    assert!(e.to_string().contains("moment shapes differ"), "{e}");
    assert_eq!(tr.steps(), 0, "a refused load changes nothing");
}

#[test]
fn a_gpu_fault_at_any_command_fails_the_step_cleanly_and_nothing_is_applied() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 41);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 4, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let mut attempts = 0;
    let commands = g.sweep_faults(|| {
        attempts += 1;
        let before = tr.steps();
        let r = tr.step(&ids, &targets, 2, &sc).map(|_| ());
        if r.is_err() {
            assert_eq!(tr.steps(), before, "a failed step does not count");
        }
        r
    });
    // The failed attempts discarded their batches, so only the final success touched the weights.
    assert!(
        commands > 100,
        "a training step issues many commands, got {commands}"
    );
    assert_eq!(attempts, commands + 1);
    assert_eq!(tr.steps(), 1);
    let fresh = GpuTrainer::new(&g, &w, None).unwrap();
    assert_eq!(
        fresh.weights_host().unwrap(),
        w,
        "sanity: the baseline weights are what we started from"
    );
    assert_ne!(
        tr.weights_host().unwrap(),
        w,
        "the one successful step updated them"
    );
}

#[test]
fn loss_and_grads_reports_a_fault_too() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 42), None).unwrap();
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let commands = g.sweep_faults(|| tr.loss_and_grads(&ids, &targets, 1).map(|_| ()));
    assert!(commands > 50, "{commands}");
}

#[test]
fn a_lost_device_surfaces_through_every_engine_method() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 43);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let snap = Engine::snapshot(&tr).unwrap();
    g.destroy();
    for _ in 0..200 {
        let _ = g.device.poll(wgpu::PollType::Poll);
        if g.check().is_err() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    assert!(Engine::step(&mut tr, &ids, &targets, 1, &sc).is_err());
    assert!(Engine::snapshot(&tr).is_err());
    assert!(Engine::restore(&mut tr, &snap).is_err());
    assert_eq!(Engine::steps(&tr), 0);
}

#[test]
fn the_engine_view_agrees_with_the_inherent_methods() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 44), None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 4, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let s = Engine::step(&mut tr, &ids, &targets, 2, &sc).unwrap();
    assert!(s.loss.is_finite());
    assert_eq!(Engine::steps(&tr), 1);
    let snap = Engine::snapshot(&tr).unwrap();
    assert_eq!(snap.steps, 1);
    assert_eq!(snap.weights, tr.weights_host().unwrap());
    Engine::restore(&mut tr, &snap).unwrap();
    assert_eq!(Engine::steps(&tr), 1);
}

fn max_err_models(a: &Gpt2, b: &Gpt2) -> f32 {
    a.params()
        .iter()
        .zip(b.params())
        .map(|(x, y)| max_rel_err(x, y))
        .fold(0.0, f32::max)
}

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

#[test]
fn split_steps_are_deterministic() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let run = || {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 55), None).unwrap();
        let sc = StepConfig {
            adam: Adam::new(0.01),
            clip: 0.5,
        };
        for s in 0..3 {
            let (ids, tg) = batch_data(&c, 4, 4, s);
            tr.step_with_micro(&ids, &tg, 4, 2, &sc).unwrap();
        }
        tr.snapshot_pair()
    };
    assert_eq!(run(), run());
}

#[test]
fn a_fault_at_any_command_of_a_split_step_applies_nothing() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 56);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let commands = g.sweep_faults(|| {
        let before = tr.steps();
        let r = tr.step_with_micro(&ids, &targets, 2, 1, &sc).map(|_| ());
        if r.is_err() {
            assert_eq!(tr.steps(), before);
        }
        r
    });
    assert!(commands > 100, "{commands}");
    assert_eq!(tr.steps(), 1);
}

/// A model wide enough for Q8_0 (every width a multiple of 32).
fn qcfg() -> Config {
    Config {
        vocab: 16,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 1,
        hidden: 64,
    }
}

fn qloop() -> TrainConfig {
    TrainConfig {
        context: 8,
        batch: 6,
        max_steps: 25,
        lr: 0.01,
        clip: 1.0,
        seed: 3,
        ..TrainConfig::default()
    }
}

#[test]
fn gpu_qat_follows_the_cpu_qat_trajectory() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let toks = pattern(400, c.vocab);
    let mut cpu = CpuTrainer::new(Gpt2::init(c, 61), 1).with_qat(true);
    let want = train(&mut cpu, &toks, &qloop(), None, None, &mut NoHooks).unwrap();
    let mut gpu = GpuTrainer::new(&g, &Gpt2::init(c, 61), None).unwrap();
    gpu.set_qat(true).unwrap();
    assert!(gpu.qat_enabled());
    let got = train(&mut gpu, &toks, &qloop(), None, None, &mut NoHooks).unwrap();
    let mut worst = 0.0f32;
    for (i, (a, b)) in got.losses.iter().zip(&want.losses).enumerate() {
        let e = (a - b).abs() / b.abs().max(1.0);
        worst = worst.max(e);
        assert!(e < 1e-4, "step {}: gpu {a} vs cpu {b}", i + 1);
    }
    eprintln!("25-step QAT loss curve: worst relative difference {worst:.2e}");
    let (first, last) = (got.losses[0], *got.losses.last().unwrap());
    assert!(last < first * 0.6, "QAT still learns: {first} -> {last}");
    // The masters stay full precision, and exporting them as Q8_0 equals the snapped model.
    let masters = gpu.weights_host().unwrap();
    assert_ne!(masters.fake_quantize(), masters);
}

#[test]
fn qat_changes_the_trajectory_and_turning_it_off_restores_the_plain_path() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let (ids, targets) = batch_data(&c, 4, 5, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let loss_of = |qat: bool| {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 62), None).unwrap();
        tr.set_qat(qat).unwrap();
        let first = tr.step(&ids, &targets, 4, &sc).unwrap().loss;
        let second = tr.step(&ids, &targets, 4, &sc).unwrap().loss;
        (first, second)
    };
    let (plain, qat) = (loss_of(false), loss_of(true));
    assert_ne!(
        plain.0, qat.0,
        "the snapped forward pass sees different weights"
    );
    assert_ne!(plain.1, qat.1);
    // On then off: the next steps are exactly the plain ones.
    let mut toggled = GpuTrainer::new(&g, &Gpt2::init(c, 62), None).unwrap();
    toggled.set_qat(true).unwrap();
    toggled.set_qat(false).unwrap();
    assert!(!toggled.qat_enabled());
    let a = toggled.step(&ids, &targets, 4, &sc).unwrap().loss;
    assert_eq!(a, plain.0);
}

#[test]
fn qat_that_does_not_fit_the_budget_is_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let w = Gpt2::init(c, 63);
    let roomy = static_plan(&c).total() + 1000;
    let mut tr = GpuTrainer::new(
        &g,
        &w,
        Some(Budget {
            bytes: roomy,
            source: Source::Override,
        }),
    )
    .unwrap();
    let before = g.buffers_created();
    let e = tr.set_qat(true).unwrap_err().to_string();
    assert!(e.contains("does not fit"), "{e}");
    assert!(!tr.qat_enabled());
    assert_eq!(g.buffers_created(), before);
    // Enabling twice is a no-op the second time.
    let mut ok = GpuTrainer::new(&g, &w, None).unwrap();
    ok.set_qat(true).unwrap();
    let made = g.buffers_created();
    ok.set_qat(true).unwrap();
    assert_eq!(g.buffers_created(), made);
}

#[test]
fn qat_micro_batches_match_the_whole_batch() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.5,
    };
    let run = |micro: usize| {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 64), None).unwrap();
        tr.set_qat(true).unwrap();
        for s in 0..3 {
            let (ids, tg) = batch_data(&c, 4, 4, s);
            tr.step_with_micro(&ids, &tg, 4, micro, &sc).unwrap();
        }
        tr.weights_host().unwrap()
    };
    let (whole, split) = (run(4), run(2));
    let e = max_err_models(&whole, &split);
    eprintln!("QAT micro-batching: weights differ by {e:.2e}");
    assert!(e < 1e-4, "{e}");
}

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

#[test]
fn load_restores_weights_moments_and_the_step_counter_exactly() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let (ids, targets) = batch_data(&c, 2, 4, 1);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    // Train 3 steps straight through.
    let mut a = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    for _ in 0..3 {
        a.step(&ids, &targets, 2, &sc).unwrap();
    }
    // Train 2, checkpoint to the host, resume in a fresh trainer for the third.
    let mut b = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    for _ in 0..2 {
        b.step(&ids, &targets, 2, &sc).unwrap();
    }
    let (w, (m, v), t) = (
        b.weights_host().unwrap(),
        b.moments_host().unwrap(),
        b.steps(),
    );
    let mut resumed = GpuTrainer::new(&g, &Gpt2::init(c, 99), None).unwrap();
    resumed.load(&w, Some((&m, &v)), t).unwrap();
    resumed.step(&ids, &targets, 2, &sc).unwrap();
    assert_eq!(resumed.steps(), 3);
    assert_eq!(
        resumed.weights_host().unwrap(),
        a.weights_host().unwrap(),
        "resume = uninterrupted"
    );
    assert_eq!(resumed.moments_host().unwrap(), a.moments_host().unwrap());
    // A different shape is refused.
    let other = Gpt2::init(Config { dim: 12, ..c }, 1);
    assert!(resumed
        .load(&other, None, 0)
        .unwrap_err()
        .to_string()
        .contains("shape"));
}

/// A learnable corpus: a repeating pattern with a few distinct tokens.
fn pattern(n: usize, vocab: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8] % vocab)
        .collect()
}

fn loop_cfg() -> TrainConfig {
    TrainConfig {
        context: 8,
        batch: 6,
        max_steps: 60,
        lr: 0.02,
        weight_decay: 0.01,
        clip: 1.0,
        warmup: 5,
        cosine: true,
        seed: 11,
        ..TrainConfig::default()
    }
}

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

struct Keep(Option<Checkpoint>, Option<u64>);
impl Hooks for Keep {
    fn on_step(&mut self, step: u64, _: &StepStats) -> bool {
        self.1 != Some(step)
    }
    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        self.0 = Some(cp.clone());
        Ok(())
    }
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

#[test]
fn two_full_gpu_runs_are_byte_identical() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let toks = pattern(300, c.vocab);
    let run = || {
        let mut t = GpuTrainer::new(&g, &Gpt2::init(c, 23), None).unwrap();
        let r = train(
            &mut t,
            &toks,
            &TrainConfig {
                max_steps: 25,
                ..loop_cfg()
            },
            None,
            None,
            &mut NoHooks,
        )
        .unwrap();
        (r.losses, t.snapshot().unwrap())
    };
    assert_eq!(run(), run());
}

#[test]
fn batch_shapes_can_change_between_steps() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 8), None).unwrap();
    let sc = StepConfig {
        adam: Adam::new(0.005),
        clip: 1.0,
    };
    for (b, t) in [(2usize, 5usize), (1, 3), (2, 5), (4, 2)] {
        let (ids, targets) = batch_data(&c, b, t, 0);
        let s = tr.step(&ids, &targets, b, &sc).unwrap();
        assert!(s.loss.is_finite() && s.grad_norm.is_finite(), "{b}x{t}");
    }
}

#[test]
fn bad_input_and_oversized_batches_are_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 9);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let mut e =
        |ids: &[usize], tg: &[usize], b: usize| tr.step(ids, tg, b, &sc).unwrap_err().to_string();
    assert!(e(&[], &[], 1).contains("split"));
    assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
    assert!(e(&[1; 9], &[1; 9], 1).contains("exceeds the context"));
    assert!(e(&[1, 2], &[1], 1).contains("targets"));
    assert!(e(&[13], &[1], 1).contains("out of range"));
    assert_eq!(tr.steps(), 0, "failed steps do not count");
    // A tight budget: the static part fits, a big batch does not.
    let need = static_plan(&c).total();
    let tight = Some(Budget {
        bytes: need + 1000,
        source: Source::Override,
    });
    let mut small = GpuTrainer::new(&g, &w, tight).unwrap();
    let before = g.buffers_created();
    let (ids, targets) = batch_data(&c, 4, 8, 0);
    let msg = small.step(&ids, &targets, 4, &sc).unwrap_err().to_string();
    assert!(
        msg.contains("does not fit") && msg.contains("budget"),
        "{msg}"
    );
    assert_eq!(
        g.buffers_created(),
        before,
        "refused before any activation buffer exists"
    );
    // And the static part alone can be refused too.
    let tiny = Some(Budget {
        bytes: 100,
        source: Source::Override,
    });
    assert!(GpuTrainer::new(&g, &w, tiny).is_err());
}
