//! A GPU run with dropout follows the CPU run with the same rate and seed.

use super::{cfg, data};
use crate::device::open_for_test;
use crate::tolerance::max_rel_err;
use crate::train::{Engine, GpuTrainer, StepConfig};
use ferrum_ref::gpt2::Gpt2;
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::CpuTrainer;

fn sc() -> StepConfig {
    StepConfig {
        adam: Adam {
            weight_decay: 0.01,
            ..Adam::new(0.01)
        },
        clip: 1.0,
    }
}

/// Which extras the pair of trainers share.
#[derive(Clone, Copy, Default)]
struct Mode {
    qat: bool,
    tie: bool,
    micro: Option<usize>,
}

/// `steps` steps on both trainers over the same batches; the worst relative loss difference and
/// the worst relative weight difference.
fn compare(mode: Mode, rate: f32, steps: usize) -> Option<(f32, f32)> {
    let g = open_for_test()?;
    let c = cfg();
    let start = Gpt2::init(c, 41);
    let mut cpu = CpuTrainer::new(start.clone(), 1)
        .with_qat(mode.qat)
        .with_tie(mode.tie)
        .with_dropout(rate, 77);
    let mut gpu = GpuTrainer::new(&g, &start, None).unwrap();
    gpu.set_qat(mode.qat).unwrap();
    gpu.set_tie(mode.tie).unwrap();
    gpu.set_dropout(rate, 77);
    let mut worst_loss = 0.0f32;
    for s in 0..steps {
        let (ids, targets) = data(&c, 4, 6, s);
        let want = cpu.step(&ids, &targets, 4, &sc()).unwrap();
        let got = match mode.micro {
            Some(m) => gpu.step_with_micro(&ids, &targets, 4, m, &sc()),
            None => gpu.step(&ids, &targets, 4, &sc()),
        }
        .unwrap();
        worst_loss = worst_loss.max((got.loss - want.loss).abs() / want.loss.abs().max(1.0));
    }
    let (a, b) = (gpu.weights_host().unwrap(), cpu.weights().clone());
    let worst_w = a
        .params()
        .iter()
        .zip(b.params())
        .fold(0.0f32, |m, (x, y)| m.max(max_rel_err(x, y)));
    Some((worst_loss, worst_w))
}

fn check(label: &str, r: Option<(f32, f32)>) {
    let Some((loss, w)) = r else { return };
    eprintln!("{label}: worst loss diff {loss:.2e}, worst weight diff {w:.2e}");
    assert!(loss < 1e-4, "{label}: loss {loss}");
    assert!(w < 1e-4, "{label}: weights {w}");
}

#[test]
fn dropout_training_follows_the_cpu_trajectory() {
    check("dropout 0.3", compare(Mode::default(), 0.3, 12));
}

#[test]
fn a_high_rate_follows_it_too() {
    check("dropout 0.9", compare(Mode::default(), 0.9, 8));
}

#[test]
fn dropout_with_qat_follows_the_cpu_trajectory() {
    check(
        "dropout+qat",
        compare(
            Mode {
                qat: true,
                ..Mode::default()
            },
            0.2,
            10,
        ),
    );
}

#[test]
fn dropout_with_a_tied_head_follows_the_cpu_trajectory() {
    check(
        "dropout+tie",
        compare(
            Mode {
                tie: true,
                ..Mode::default()
            },
            0.2,
            10,
        ),
    );
}

#[test]
fn micro_batched_dropout_drops_the_same_units_as_the_whole_batch() {
    // the units of micro-batch c are numbered from the start of the batch, so the split run is
    // the CPU's whole-batch run up to summation order
    check(
        "dropout, 2 micro-batches",
        compare(
            Mode {
                micro: Some(2),
                ..Mode::default()
            },
            0.3,
            8,
        ),
    );
    check(
        "dropout, 4 micro-batches",
        compare(
            Mode {
                micro: Some(1),
                ..Mode::default()
            },
            0.3,
            6,
        ),
    );
}

#[test]
fn a_gpu_loss_and_gradient_pass_with_dropout_matches_the_cpu_pass() {
    use ferrum_ref::dropout::{step_seed, Dropout};
    use ferrum_ref::grad::loss_and_grads_with;
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 5);
    let (ids, targets) = data(&c, 2, 5, 0);
    let mut gpu = GpuTrainer::new(&g, &w, None).unwrap();
    gpu.set_dropout(0.4, 3);
    let loss = gpu.loss_and_grads(&ids, &targets, 2).unwrap();
    let d = Dropout::new(0.4, step_seed(3, 0));
    let (want_loss, want) = loss_and_grads_with(&w, &ids, &targets, 2, 1, d).unwrap();
    assert!(
        (loss - want_loss).abs() < 1e-5 * want_loss.max(1.0),
        "{loss} vs {want_loss}"
    );
    let got = gpu.grads_host().unwrap();
    for (i, (a, b)) in got.params().iter().zip(want.params()).enumerate() {
        let e = max_rel_err(a, b);
        assert!(e < crate::tolerance::BACKWARD, "param {i}: {e}");
    }
}
