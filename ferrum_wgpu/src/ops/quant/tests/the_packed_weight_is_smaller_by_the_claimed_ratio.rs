use super::*;

#[test]
fn the_packed_weight_is_smaller_by_the_claimed_ratio() {
    let Some(g) = open_for_test() else { return };
    let (k, n) = (64, 32);
    for (kind, ratio) in [
        (QKind::Q8_0, 128.0 / 36.0),
        (QKind::Q4_0, 128.0 / 20.0),
        (QKind::F16, 2.0),
    ] {
        let qw = upload_quantised(&g, kind, &rand_vec(k * n, 1), k, n).unwrap();
        let f32_bytes = (k * n * 4) as f64;
        assert!(
            (f32_bytes / qw.device_bytes() as f64 - ratio).abs() < 1e-9,
            "{kind:?}"
        );
        assert_eq!((qw.k(), qw.n(), qw.kind()), (k, n, kind));
    }
}

#[test]
fn linear_refuses_mismatched_buffers_and_fails_cleanly() {
    let Some(g) = open_for_test() else { return };
    let qw = upload_quantised(&g, QKind::Q8_0, &rand_vec(64 * 4, 1), 64, 4).unwrap();
    let (x, y) = (g.zeros(2 * 64).unwrap(), g.zeros(2 * 4).unwrap());
    assert!(linear_q(&g, &x, &qw, &y, 3)
        .unwrap_err()
        .to_string()
        .contains("linear_q x"));
    assert!(linear_q(&g, &x, &qw, &g.zeros(7).unwrap(), 2)
        .unwrap_err()
        .to_string()
        .contains("linear_q y"));
    assert_eq!(g.sweep_faults(|| linear_q(&g, &x, &qw, &y, 2)), 1);
    assert!(upload_quantised(&g, QKind::Q4_0, &[0.0; 33], 33, 1).is_err());
}

/// Naive against the fast kernels, per precision, at the shapes of a decode step (one row) and of
/// prefill (32 rows). Ignored by default: it prints timings, it asserts nothing about speed.
/// `cargo test --release -p ferrum_wgpu --lib -- --ignored bench_quant --nocapture`
#[test]
#[ignore]
fn bench_quant_paths() {
    use crate::bench_util::time_ms;
    let Some(g) = open_for_test() else { return };
    eprintln!("adapter: {}", g.info().name);
    // every width is a whole number of Q2_K super-blocks too
    for (k, n) in [(512usize, 512usize), (512, 1408), (1024, 512), (512, 2048)] {
        for kind in KINDS.into_iter().chain(super::ext_tests::EXT) {
            let w = rand_vec(k * n, 1);
            let qw = upload_quantised(&g, kind, &w, k, n).unwrap();
            for rows in [1usize, 32] {
                let gx = g.upload(&rand_vec(rows * k, 2)).unwrap();
                let gy = g.zeros(rows * n).unwrap();
                let fast = if rows == 1 { QPath::Gemv } else { QPath::Tiled };
                let mut t = [0.0f64; 2];
                for (i, path) in [QPath::Naive, fast].into_iter().enumerate() {
                    // 20 launches per timed run, then one read to wait for them
                    t[i] = time_ms(9, || {
                        for _ in 0..20 {
                            linear_q_with(&g, &gx, &qw, &gy, rows, path).unwrap();
                        }
                        g.read(&gy).unwrap();
                    }) / 20.0;
                }
                eprintln!(
                    "k={k:5} n={n:5} rows={rows:2} {kind:?}: naive {:.3} ms, fast {:.3} ms ({:+.0}%)",
                    t[0],
                    t[1],
                    (t[1] / t[0] - 1.0) * 100.0
                );
            }
        }
    }
}
