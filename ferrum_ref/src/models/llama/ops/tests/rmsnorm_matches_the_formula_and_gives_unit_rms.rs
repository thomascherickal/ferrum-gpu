use super::*;

#[test]
fn rmsnorm_matches_the_formula_and_gives_unit_rms() {
    let x = [3.0, 4.0];
    let y = rmsnorm(&x, &[1.0, 2.0], 1, 2, 0.0);
    let inv = 1.0 / (12.5f32).sqrt();
    assert!((y[0] - 3.0 * inv).abs() < 1e-6 && (y[1] - 8.0 * inv).abs() < 1e-6);
    let r = rand_vec(3 * 8, 1);
    let y = rmsnorm(&r, &[1.0; 8], 3, 8, 0.0);
    for row in y.chunks(8) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
        assert!((ms - 1.0).abs() < 1e-4);
    }
}

#[test]
fn silu_and_swiglu_known_values() {
    assert_eq!(silu(0.0), 0.0);
    assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
    assert!((silu(-20.0)).abs() < 1e-6 && (silu(20.0) - 20.0).abs() < 1e-5);
    let h = swiglu(&[1.0, 0.0], &[2.0, 5.0]);
    assert!((h[0] - 2.0 * silu(1.0)).abs() < 1e-6 && h[1] == 0.0);
}

#[test]
fn rope_at_position_zero_is_the_identity() {
    let (c, s) = rope_tables(0, 1, 4, 10000.0, &RopeScaling::default());
    assert_eq!((c, s.clone()), (vec![1.0, 1.0], vec![0.0, 0.0]));
    let mut x = rand_vec(8, 2);
    let orig = x.clone();
    let (c, s) = rope_tables(0, 1, 4, 10000.0, &RopeScaling::default());
    apply_rope(&mut x, (1, 2, 4, 4), (&c, &s), RopeType::Norm);
    assert_eq!(x, orig);
}

#[test]
fn rope_preserves_pair_norms_and_the_two_pairings_differ() {
    let x0 = rand_vec(8, 3);
    let (c, s) = rope_tables(5, 1, 8, 10000.0, &RopeScaling::default());
    let (mut n, mut e) = (x0.clone(), x0.clone());
    apply_rope(&mut n, (1, 1, 8, 8), (&c, &s), RopeType::Norm);
    apply_rope(&mut e, (1, 1, 8, 8), (&c, &s), RopeType::Neox);
    for i in 0..4 {
        let before = x0[2 * i].powi(2) + x0[2 * i + 1].powi(2);
        let after = n[2 * i].powi(2) + n[2 * i + 1].powi(2);
        assert!((before - after).abs() < 1e-5);
        let before = x0[i].powi(2) + x0[i + 4].powi(2);
        let after = e[i].powi(2) + e[i + 4].powi(2);
        assert!((before - after).abs() < 1e-5);
    }
    assert_ne!(n, e);
}

#[test]
fn rope_only_rotates_the_first_rope_dim_channels() {
    let mut x = rand_vec(8, 4);
    let orig = x.clone();
    let (c, s) = rope_tables(3, 1, 4, 10000.0, &RopeScaling::default());
    apply_rope(&mut x, (1, 1, 8, 4), (&c, &s), RopeType::Neox);
    assert_eq!(x[4..], orig[4..]);
    assert_ne!(x[..4], orig[..4]);
}
