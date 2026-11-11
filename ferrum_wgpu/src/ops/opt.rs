//! Optimizer kernels: deterministic global gradient norm, clip scale, and the AdamW update.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::optim::Adam;

/// Elements summarised by one workgroup of the sum-of-squares kernel.
pub const CHUNK: usize = 4096;

/// Partial sums a tensor of `n` elements produces.
pub fn parts_for(n: usize) -> usize {
    n.div_ceil(CHUNK).max(1)
}

/// The sum-of-squares kernel runs one workgroup per partial along x, so `partials` must fit.
/// (Split out so the limit is testable without a 1 GB tensor.)
fn check_dispatchable(partials: usize) -> Result<(), GpuError> {
    if partials > crate::runtime::GROUPS_X {
        return Err(GpuError(
            "sumsq_parts: tensor too large for one dispatch".into(),
        ));
    }
    Ok(())
}

fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(GpuError(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

/// Write `parts_for(g.len())` partial sums of squares of `g` into `part[offset..]`.
pub fn sumsq_parts(gpu: &Gpu, g: &GpuBuf, part: &GpuBuf, offset: usize) -> Result<(), GpuError> {
    let k = parts_for(g.len());
    if offset + k > part.len() {
        return Err(GpuError(format!(
            "sumsq_parts: {k} partials at {offset} exceed the buffer of {}",
            part.len()
        )));
    }
    check_dispatchable(k)?;
    let p = gpu.pipeline("sumsq_part", include_str!("../shaders/opt/sumsq_part.wgsl"))?;
    gpu.dispatch(
        &p,
        &[g, part],
        &[g.len() as u32, offset as u32],
        (k as u32, 1),
    )
}

/// `out = [clip scale, global norm]` from the first `total` partials.
pub fn norm_scale(
    gpu: &Gpu,
    part: &GpuBuf,
    out: &GpuBuf,
    total: usize,
    max_norm: f32,
) -> Result<(), GpuError> {
    need("norm_scale out", out, 2)?;
    if total > part.len() {
        return Err(GpuError(format!(
            "norm_scale: {total} partials exceed the buffer of {}",
            part.len()
        )));
    }
    let p = gpu.pipeline("norm_final", include_str!("../shaders/opt/norm_final.wgsl"))?;
    gpu.dispatch(
        &p,
        &[part, out],
        &[total as u32, max_norm.to_bits()],
        (1, 1),
    )
}

/// `acc = (first ? 0 : acc) + weight * g`: add one micro-batch's gradient into the running sum.
pub fn accumulate(
    gpu: &Gpu,
    acc: &GpuBuf,
    g: &GpuBuf,
    weight: f32,
    first: bool,
) -> Result<(), GpuError> {
    need("accumulate g", g, acc.len())?;
    let p = gpu.pipeline("accumulate", include_str!("../shaders/opt/accumulate.wgsl"))?;
    gpu.dispatch_1d(
        &p,
        &[acc, g],
        &[acc.len() as u32, weight.to_bits(), u32::from(first)],
        acc.len(),
    )
}

/// One AdamW update of a tensor. `decay` selects whether weight decay applies; `t` is the
/// 1-based step (bias corrections are computed here, on the host). `scale` is `[clip scale, ..]`.
pub fn adam_update(
    gpu: &Gpu,
    (p, g, m, v): (&GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf),
    scale: &GpuBuf,
    adam: &Adam,
    decay: bool,
    t: u64,
) -> Result<(), GpuError> {
    let n = p.len();
    for (name, b) in [("grad", g), ("m", m), ("v", v)] {
        need(&format!("adam_update {name}"), b, n)?;
    }
    if t == 0 {
        return Err(GpuError("adam timestep must be at least 1".into()));
    }
    if scale.is_empty() {
        return Err(GpuError("adam_update: scale buffer is empty".into()));
    }
    let (bc1, bc2) = adam.bias_corrections(t);
    let wd = if decay { adam.weight_decay } else { 0.0 };
    let u = [
        n as u32,
        adam.lr.to_bits(),
        adam.beta1.to_bits(),
        adam.beta2.to_bits(),
        adam.eps.to_bits(),
        wd.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let pipe = gpu.pipeline("adam", include_str!("../shaders/opt/adam.wgsl"))?;
    gpu.dispatch_1d(&pipe, &[p, g, m, v, scale], &u, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use crate::tolerance::max_rel_err;
    use ferrum_ref::gpt2::{Config, Gpt2};
    use ferrum_ref::matmul::rand_vec;
    use ferrum_ref::optim::{adam_step, clip_scale, global_norm, AdamState};

    #[test]
    fn the_dispatch_limit_is_enforced_without_needing_a_huge_tensor() {
        assert!(check_dispatchable(1).is_ok());
        assert!(check_dispatchable(crate::runtime::GROUPS_X).is_ok());
        let e = check_dispatchable(crate::runtime::GROUPS_X + 1).unwrap_err();
        assert!(e.to_string().contains("too large"));
        // The biggest tensor that fits: partials * CHUNK elements.
        assert_eq!(
            parts_for(crate::runtime::GROUPS_X * CHUNK),
            crate::runtime::GROUPS_X
        );
    }

    #[test]
    fn accumulate_builds_a_weighted_sum_and_first_resets_stale_contents() {
        let Some(g) = open_for_test() else { return };
        let n = 5000; // more than one 64-wide workgroup row of work
        let acc = g.upload(&vec![9.0; n]).unwrap(); // stale data that `first` must discard
        let parts: Vec<Vec<f32>> = (0..3).map(|s| rand_vec(n, 50 + s)).collect();
        let weights = [0.5f32, 0.25, 0.125];
        for (i, (p, w)) in parts.iter().zip(weights).enumerate() {
            accumulate(&g, &acc, &g.upload(p).unwrap(), w, i == 0).unwrap();
        }
        let mut want = vec![0.0f32; n];
        for (p, w) in parts.iter().zip(weights) {
            for (a, x) in want.iter_mut().zip(p) {
                *a += w * x;
            }
        }
        assert_eq!(
            g.read(&acc).unwrap(),
            want,
            "same operations in the same order"
        );
        let e = accumulate(&g, &acc, &g.zeros(3).unwrap(), 1.0, false).unwrap_err();
        assert!(e.to_string().contains("accumulate g"));
        assert_eq!(
            g.sweep_faults(|| accumulate(&g, &acc, &acc_src(&g, n), 1.0, true)),
            1
        );
    }

    fn acc_src(g: &Gpu, n: usize) -> GpuBuf {
        g.zeros(n).unwrap()
    }

    #[test]
    fn an_empty_scale_buffer_is_refused() {
        let Some(g) = open_for_test() else { return };
        let a = g.zeros(4).unwrap();
        let empty = g.zeros(0).unwrap();
        let e = adam_update(&g, (&a, &a, &a, &a), &empty, &Adam::new(0.1), true, 1).unwrap_err();
        assert!(e.to_string().contains("scale buffer is empty"));
    }

    #[test]
    fn parts_for_counts_whole_chunks() {
        assert_eq!(parts_for(0), 1);
        assert_eq!(parts_for(1), 1);
        assert_eq!(parts_for(CHUNK), 1);
        assert_eq!(parts_for(CHUNK + 1), 2);
        assert_eq!(parts_for(10 * CHUNK), 10);
    }

    #[test]
    fn global_norm_and_clip_scale_match_cpu_across_chunk_boundaries() {
        let Some(g) = open_for_test() else { return };
        for sizes in [
            vec![1usize],
            vec![4095, 1],
            vec![4096, 4097],
            vec![100_000, 3, 70_001],
        ] {
            let tensors: Vec<Vec<f32>> = sizes
                .iter()
                .enumerate()
                .map(|(i, &n)| rand_vec(n, 30 + i as u64))
                .collect();
            let total: usize = tensors.iter().map(|t| parts_for(t.len())).sum();
            let part = g.zeros(total).unwrap();
            let mut off = 0;
            for t in &tensors {
                let b = g.upload(t).unwrap();
                sumsq_parts(&g, &b, &part, off).unwrap();
                off += parts_for(t.len());
            }
            let want: f64 = tensors
                .iter()
                .flatten()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum();
            let want_norm = want.sqrt() as f32;
            for max in [0.0f32, 1.0, want_norm * 2.0, want_norm / 4.0] {
                let out = g.zeros(2).unwrap();
                norm_scale(&g, &part, &out, total, max).unwrap();
                let got = g.read(&out).unwrap();
                let (scale, norm) = (got[0], got[1]);
                assert!(
                    (norm - want_norm).abs() < 1e-5 * want_norm.max(1.0),
                    "norm {norm} vs {want_norm} for {sizes:?}"
                );
                let want_scale = clip_scale(want_norm, max);
                assert!(
                    (scale - want_scale).abs() < 1e-5,
                    "scale {scale} vs {want_scale} (max {max})"
                );
            }
        }
    }

    #[test]
    fn a_non_finite_gradient_is_never_clipped_into_a_finite_scale() {
        let Some(g) = open_for_test() else { return };
        let bad = g.upload(&[1.0, f32::NAN, 2.0]).unwrap();
        let part = g.zeros(1).unwrap();
        sumsq_parts(&g, &bad, &part, 0).unwrap();
        let out = g.zeros(2).unwrap();
        norm_scale(&g, &part, &out, 1, 1.0).unwrap();
        let r = g.read(&out).unwrap();
        assert_eq!(r[0], 1.0, "NaN norm leaves the scale at 1");
        assert!(r[1].is_nan());
    }

    #[test]
    fn norm_is_bit_identical_across_runs() {
        let Some(g) = open_for_test() else { return };
        let data = rand_vec(50_000, 41);
        let run = || {
            let b = g.upload(&data).unwrap();
            let part = g.zeros(parts_for(data.len())).unwrap();
            sumsq_parts(&g, &b, &part, 0).unwrap();
            let out = g.zeros(2).unwrap();
            norm_scale(&g, &part, &out, parts_for(data.len()), 0.0).unwrap();
            g.read(&out).unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn adam_update_matches_the_cpu_over_several_steps_with_and_without_decay() {
        let Some(g) = open_for_test() else { return };
        let cfg = Config {
            vocab: 9,
            ctx: 6,
            dim: 8,
            heads: 2,
            layers: 1,
            hidden: 12,
        };
        let w0 = Gpt2::random(cfg, 7);
        let adam = Adam {
            weight_decay: 0.05,
            ..Adam::new(0.01)
        };
        let mut cpu = w0.clone();
        let mut st = AdamState::new(&w0);
        let up = |m: &Gpt2| {
            m.params()
                .iter()
                .map(|p| g.upload(p).unwrap())
                .collect::<Vec<_>>()
        };
        let (gp, gm, gv) = (up(&w0), up(&w0.zeros_like()), up(&w0.zeros_like()));
        let scale = g.upload(&[0.5, 0.0]).unwrap();
        for step in 1..=4u64 {
            let grads = Gpt2::random(cfg, 100 + step);
            adam_step(&mut cpu, &grads, &mut st, &adam, 0.5);
            let gg = up(&grads);
            for i in 0..gp.len() {
                adam_update(
                    &g,
                    (&gp[i], &gg[i], &gm[i], &gv[i]),
                    &scale,
                    &adam,
                    w0.decays(i),
                    step,
                )
                .unwrap();
            }
        }
        for (i, want) in cpu.params().iter().enumerate() {
            let got = g.read(&gp[i]).unwrap();
            let e = max_rel_err(&got, want);
            assert!(e < 1e-5, "param {i}: max rel err {e}");
        }
        for (i, want) in st.v.params().iter().enumerate() {
            assert!(max_rel_err(&g.read(&gv[i]).unwrap(), want) < 1e-5, "v {i}");
        }
        let _ = global_norm(&cpu);
    }

    #[test]
    fn errors_are_reported_not_dispatched() {
        let Some(g) = open_for_test() else { return };
        let (a, b) = (g.zeros(4).unwrap(), g.zeros(3).unwrap());
        let s = g.zeros(2).unwrap();
        assert!(adam_update(&g, (&a, &b, &a, &a), &s, &Adam::new(0.1), true, 1).is_err());
        assert!(
            adam_update(&g, (&a, &a, &a, &a), &s, &Adam::new(0.1), true, 0)
                .unwrap_err()
                .to_string()
                .contains("at least 1")
        );
        let part = g.zeros(1).unwrap();
        assert!(
            sumsq_parts(&g, &a, &part, 1).is_err(),
            "offset beyond the buffer"
        );
        assert!(norm_scale(&g, &part, &g.zeros(1).unwrap(), 1, 1.0).is_err());
        assert!(norm_scale(&g, &part, &s, 5, 1.0).is_err());
    }
}
