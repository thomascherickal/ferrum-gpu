use super::*;

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
