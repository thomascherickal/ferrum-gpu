//! The GPU tied trainer follows the CPU tied trainer step for step.

use super::{cfg, data, is_tied};
use crate::device::open_for_test;
use crate::tolerance::max_rel_err;
use crate::train::{Engine, GpuTrainer, StepConfig};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::CpuTrainer;

fn step_config() -> StepConfig {
    StepConfig {
        adam: Adam {
            weight_decay: 0.01,
            ..Adam::new(0.01)
        },
        clip: 1.0,
    }
}

/// Run `steps` steps on both trainers over the same batches; the worst relative loss difference
/// and the worst relative weight difference.
fn compare(qat: bool, steps: usize, micro: Option<usize>) -> Option<(f32, f32)> {
    let g = open_for_test()?;
    let c = cfg();
    let start = Gpt2::init(c, 41);
    let mut cpu = CpuTrainer::new(start.clone(), 1)
        .with_tie(true)
        .with_qat(qat);
    let mut gpu = GpuTrainer::new(&g, &start, None).unwrap();
    gpu.set_qat(qat).unwrap();
    gpu.set_tie(true).unwrap();
    let (mut worst_loss, mut worst_w) = (0.0f32, 0.0f32);
    for s in 0..steps {
        let (ids, targets) = data(&c, 4, 6, s);
        let want = cpu.step(&ids, &targets, 4, &step_config()).unwrap();
        let got = match micro {
            Some(m) => gpu.step_with_micro(&ids, &targets, 4, m, &step_config()),
            None => gpu.step(&ids, &targets, 4, &step_config()),
        }
        .unwrap();
        worst_loss = worst_loss.max((got.loss - want.loss).abs() / want.loss.abs().max(1.0));
    }
    let (a, b) = (gpu.weights_host().unwrap(), cpu.weights().clone());
    assert!(
        is_tied(&a),
        "the GPU head is the embedding transposed, exactly"
    );
    for (x, y) in a.params().iter().zip(b.params()) {
        worst_w = worst_w.max(max_rel_err(x, y));
    }
    Some((worst_loss, worst_w))
}

fn check(label: &str, r: Option<(f32, f32)>, bound: f32) {
    let Some((loss, w)) = r else { return };
    eprintln!("{label}: worst loss diff {loss:.2e}, worst weight diff {w:.2e}");
    assert!(loss < bound, "{label}: loss {loss}");
    assert!(w < bound, "{label}: weights {w}");
}

#[test]
fn tied_training_follows_the_cpu_trajectory() {
    check("tied", compare(false, 12, None), 1e-4);
}

#[test]
fn tied_qat_training_follows_the_cpu_trajectory() {
    check("tied+qat", compare(true, 12, None), 1e-4);
}

#[test]
fn micro_batched_tied_steps_match_the_cpu_too() {
    // the accumulated gradients are folded, not the per-chunk ones
    check("tied, 2 micro-batches", compare(false, 8, Some(2)), 1e-4);
}

#[test]
fn tying_changes_the_run() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let start = Gpt2::init(c, 41);
    let (ids, targets) = data(&c, 4, 6, 0);
    let mut tied = GpuTrainer::new(&g, &start, None).unwrap();
    tied.set_tie(true).unwrap();
    let mut plain = GpuTrainer::new(&g, &start, None).unwrap();
    let a: Vec<f32> = (0..3)
        .map(|_| tied.step(&ids, &targets, 4, &step_config()).unwrap().loss)
        .collect();
    let b: Vec<f32> = (0..3)
        .map(|_| plain.step(&ids, &targets, 4, &step_config()).unwrap().loss)
        .collect();
    assert_ne!(a, b);
}
