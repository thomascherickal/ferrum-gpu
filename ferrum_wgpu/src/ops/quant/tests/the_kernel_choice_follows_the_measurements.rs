use super::*;

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
