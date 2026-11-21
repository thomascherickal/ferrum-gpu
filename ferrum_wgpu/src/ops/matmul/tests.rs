use super::*;
use crate::device::open_for_test;
use ferrum_ref::matmul::{matmul, rand_vec};

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
        .fold(0.0, f32::max)
}

#[test]
fn dispatch_size_covers_the_output_and_enforces_the_limit() {
    assert_eq!(dispatch_size(Variant::Naive, 9, 17, 65535).unwrap(), (3, 2));
    assert_eq!(
        dispatch_size(Variant::Tiled16, 16, 16, 65535).unwrap(),
        (1, 1)
    );
    assert_eq!(
        dispatch_size(Variant::Blocked4, 65, 1, 65535).unwrap(),
        (1, 2)
    );
    let e = dispatch_size(Variant::Naive, 8 * 70_000, 8, 65535).unwrap_err();
    assert!(e.to_string().contains("65535"));
}

#[test]
fn every_variant_matches_the_cpu_reference_on_odd_and_degenerate_shapes() {
    let Some(g) = open_for_test() else { return };
    let shapes = [
        (1, 1, 1),
        (1, 7, 5),
        (5, 1, 9),
        (17, 33, 15),
        (64, 64, 64),
        (65, 17, 63),
        (3, 129, 130),
        (100, 3, 1),
    ];
    for v in Variant::ALL {
        for &(m, k, n) in &shapes {
            let a = rand_vec(m * k, 11);
            let b = rand_vec(k * n, 12);
            let want = matmul(&a, &b, m, k, n).unwrap();
            let got = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
            let err = max_rel_err(&got, &want);
            assert!(err < 1e-4, "{v:?} {m}x{k}x{n}: max rel err {err}");
        }
    }
}

#[test]
fn repeat_runs_are_bit_identical() {
    let Some(g) = open_for_test() else { return };
    let (m, k, n) = (70, 50, 90);
    let a = rand_vec(m * k, 5);
    let b = rand_vec(k * n, 6);
    for v in Variant::ALL {
        let first = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
        let second = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
        assert_eq!(first, second, "{v:?}");
    }
}

#[test]
fn roundtrip_and_wait_succeed_on_an_idle_device() {
    let Some(g) = open_for_test() else { return };
    g.roundtrip().unwrap();
    g.wait().unwrap();
}

#[test]
fn mismatched_buffers_are_rejected() {
    let Some(g) = open_for_test() else { return };
    let mm = Matmul::new(&g, Variant::Naive).unwrap();
    let (a, b, c) = (
        g.zeros(4).unwrap(),
        g.zeros(4).unwrap(),
        g.zeros(3).unwrap(),
    );
    let e = mm.submit(&g, &a, &b, &c, (2, 2, 2)).unwrap_err();
    assert!(e.to_string().contains("do not match"));
}
