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

mod a_gpu_loss_and_gradient_pass_with_dropout_matches_the_cpu_pass;
mod dropout_training_follows_the_cpu_trajectory;
