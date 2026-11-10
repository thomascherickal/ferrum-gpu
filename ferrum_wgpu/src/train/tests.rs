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
