use super::*;
use crate::matmul::rand_vec;

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

#[test]
fn rope_scaling_divides_frequencies_and_positions_by_hand() {
    let sc = RopeScaling {
        freq_factors: vec![2.0, 1.0],
        linear_factor: 4.0,
    };
    let (c, s) = rope_tables(8, 1, 4, 100.0, &sc);
    let f1 = 100f32.powf(-0.5); // frequency 1, rope_dim 4
    let want0 = 8.0f32 * 1.0 / 2.0 / 4.0;
    let want1 = 8.0f32 * f1 / 4.0;
    assert!((c[0] - want0.cos()).abs() < 1e-6 && (s[0] - want0.sin()).abs() < 1e-6);
    assert!((c[1] - want1.cos()).abs() < 1e-6 && (s[1] - want1.sin()).abs() < 1e-6);
    // unit factors are bit-identical to no scaling
    let unit = RopeScaling {
        freq_factors: vec![1.0, 1.0],
        linear_factor: 1.0,
    };
    assert_eq!(
        rope_tables(7, 3, 4, 1e4, &unit),
        rope_tables(7, 3, 4, 1e4, &RopeScaling::default())
    );
}

#[test]
fn rope_scaling_check_refuses_unusable_factors() {
    let bad = |f: Vec<f32>, l: f32| RopeScaling {
        freq_factors: f,
        linear_factor: l,
    };
    assert!(RopeScaling::default().check(8).is_ok());
    assert!(bad(vec![1.0; 4], 2.0).check(8).is_ok());
    assert!(bad(vec![1.0; 3], 1.0)
        .check(8)
        .unwrap_err()
        .contains("expected"));
    assert!(bad(vec![], 0.0).check(8).is_err());
    assert!(bad(vec![-1.0; 4], 1.0).check(8).is_err());
    assert!(bad(vec![f32::NAN; 4], 1.0).check(8).is_err());
}

#[test]
fn attention_with_one_position_returns_v_and_rows_are_convex() {
    let (heads, kv, hd) = (4, 2, 3);
    let q = rand_vec(heads * hd, 5);
    let k = rand_vec(kv * hd, 6);
    let v = rand_vec(kv * hd, 7);
    let o = attend_row(&q, &k, &v, 1, (heads, kv, hd));
    for h in 0..heads {
        let g = (h / 2) * hd;
        assert_eq!(&o[h * hd..(h + 1) * hd], &v[g..g + hd], "head {h}");
    }
}

#[test]
fn gqa_shares_kv_heads_between_grouped_query_heads() {
    // Two query heads with identical q and one KV head must produce identical outputs.
    let hd = 4;
    let q1 = rand_vec(hd, 8);
    let q: Vec<f32> = [q1.clone(), q1].concat();
    let (k, v) = (rand_vec(3 * hd, 9), rand_vec(3 * hd, 10));
    let o = attend_row(&q, &k, &v, 3, (2, 1, hd));
    assert_eq!(o[..hd], o[hd..]);
    // With kv_heads == heads each head uses its own K/V.
    let k2 = rand_vec(3 * 2 * hd, 11);
    let v2 = rand_vec(3 * 2 * hd, 12);
    let o2 = attend_row(&q, &k2, &v2, 3, (2, 2, hd));
    assert_ne!(o2[..hd], o2[hd..]);
}
