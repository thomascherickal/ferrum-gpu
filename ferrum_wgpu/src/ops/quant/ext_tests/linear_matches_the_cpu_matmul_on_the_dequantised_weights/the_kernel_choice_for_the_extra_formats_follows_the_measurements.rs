use super::*;

#[test]
pub(super) fn the_kernel_choice_for_the_extra_formats_follows_the_measurements() {
    // several rows: tiled
    for kind in EXT {
        assert_eq!(choose(32, kind, 512, 1408), QPath::Tiled);
        assert_eq!(choose(2, kind, 256, 3), QPath::Tiled);
    }
    // one row: per-column unless the output is wider than the input, where MXFP4 and FP8 are
    // faster naive on the MX330; Q2_K always per-column
    for kind in [QKind::Fp4, QKind::Fp8E4M3, QKind::Fp8E5M2] {
        assert_eq!(choose(1, kind, 512, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 1024, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 512, 1408), QPath::Naive);
    }
    for (k, n) in [(512, 512), (1024, 512), (512, 1408), (512, 2048)] {
        assert_eq!(choose(1, QKind::Q2K, k, n), QPath::Gemv);
    }
}

#[test]
pub(super) fn the_per_column_kernel_still_refuses_more_than_one_row_for_the_extra_formats() {
    let Some(g) = open_for_test() else { return };
    let qw = upload_quantised(&g, QKind::Fp4, &rand_vec(32 * 4, 2), 32, 4).unwrap();
    let (gx, gy) = (g.upload(&rand_vec(64, 3)).unwrap(), g.zeros(8).unwrap());
    assert!(linear_q_with(&g, &gx, &qw, &gy, 2, QPath::Gemv).is_err());
}

#[test]
pub(super) fn the_quantised_product_stays_near_the_f32_product() {
    let Some(g) = open_for_test() else { return };
    // loose bounds that only fail if a format is decoded wrongly: 2-bit values are coarse
    for (kind, k, bound) in [
        (QKind::Fp4, 128, 0.35),
        (QKind::Fp8E4M3, 128, 0.04),
        (QKind::Fp8E5M2, 128, 0.08),
        (QKind::Q2K, 256, 0.9),
    ] {
        let (n, rows) = (24, 5);
        let w = rand_vec(k * n, 4);
        let x = rand_vec(rows * k, 6);
        let want = matmul(&x, &w, rows, k, n).unwrap();
        let qw = upload_quantised(&g, kind, &w, k, n).unwrap();
        let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
        linear_q(&g, &gx, &qw, &gy, rows).unwrap();
        let got = g.read(&gy).unwrap();
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(worst <= bound * scale, "{kind:?}: {worst} vs {scale}");
    }
}
