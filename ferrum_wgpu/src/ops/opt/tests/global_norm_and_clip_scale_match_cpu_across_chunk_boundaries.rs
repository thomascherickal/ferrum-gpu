use super::*;

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
