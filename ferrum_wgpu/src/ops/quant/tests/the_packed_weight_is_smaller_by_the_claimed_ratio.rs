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

#[test]
fn the_kernel_choice_follows_the_measurements() {
    // one row: the per-column kernel unless the output is wider than the input
    for kind in KINDS {
        assert_eq!(choose(1, kind, 512, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 1408, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 512, 1408), QPath::Naive);
        assert_eq!(choose(1, kind, 512, 2048), QPath::Naive);
    }
    // several rows: tiled, except Q4_0 where it measured no better
    assert_eq!(choose(32, QKind::Q8_0, 512, 1408), QPath::Tiled);
    assert_eq!(choose(2, QKind::F16, 64, 3), QPath::Tiled);
    assert_eq!(choose(32, QKind::Q4_0, 512, 1408), QPath::Naive);
}

#[test]
fn every_kernel_matches_the_cpu_matmul_on_every_precision_and_awkward_shapes() {
    let Some(g) = open_for_test() else { return };
    // n and rows that are not multiples of the tile, n of one, k of one block and of many
    for kind in KINDS {
        for (rows, k, n) in [
            (1usize, 32usize, 1usize),
            (1, 96, 17),
            (1, 512, 70),
            (17, 64, 1),
            (33, 96, 17),
            (16, 32, 16),
            (3, 256, 40),
        ] {
            let w = rand_vec(k * n, 6);
            let bytes = quantise_matrix(kind, &w, k, n).unwrap();
            let want_w = transpose(&decoded_rows(kind, &bytes, k, n), n, k);
            let x = rand_vec(rows * k, 9);
            let want = matmul(&x, &want_w, rows, k, n).unwrap();
            let qw = upload_qweight(&g, kind, &bytes, k, n).unwrap();
            let paths: &[QPath] = if rows == 1 {
                &[QPath::Naive, QPath::Tiled, QPath::Gemv, QPath::Auto]
            } else {
                &[QPath::Naive, QPath::Tiled, QPath::Auto]
            };
            for &path in paths {
                let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
                linear_q_with(&g, &gx, &qw, &gy, rows, path).unwrap();
                let e = max_rel_err(&g.read(&gy).unwrap(), &want);
                assert!(e < REDUCING, "{kind:?} {path:?} {rows}x{k}x{n}: {e}");
            }
        }
    }
}

#[test]
fn the_per_column_kernel_refuses_more_than_one_row() {
    let Some(g) = open_for_test() else { return };
    let qw = upload_quantised(&g, QKind::Q8_0, &rand_vec(64 * 4, 1), 64, 4).unwrap();
    let (gx, gy) = (g.zeros(2 * 64).unwrap(), g.zeros(2 * 4).unwrap());
    let e = linear_q_with(&g, &gx, &qw, &gy, 2, QPath::Gemv)
        .unwrap_err()
        .to_string();
    assert!(e.contains("one row"), "{e}");
}
