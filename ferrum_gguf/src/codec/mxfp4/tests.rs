use super::*;

#[test]
fn the_half_scale_is_two_to_the_e_minus_128_for_every_byte() {
    for e in 0..=254u8 {
        assert_eq!(half_scale(e), 2f64.powi(i32::from(e) - 128) as f32, "e={e}");
    }
}

#[test]
fn a_block_of_representable_values_round_trips_exactly() {
    // 1.5 * 2^k scale: the values are kvalues * 2^(e-128); pick e = 127 (d = 0.5)
    let vals: Vec<f32> = (0..32).map(|i| f32::from(VALUES[i % 16]) * 0.5).collect();
    let bytes = encode(&vals);
    assert_eq!(bytes.len(), BLOCK);
    // amax = 6, floor(log2 6) = 2, e = 2 - 2 + 127 = 127
    assert_eq!(bytes[0], 127);
    // the low nibble of byte j is element j, the high nibble element j + 16
    assert_eq!(bytes[1] & 15, 0);
    assert_eq!(bytes[2] & 15, 1);
    assert_eq!(decode(&bytes), vals);
}

#[test]
fn the_error_is_at_most_half_a_step_inside_the_range() {
    let data: Vec<f32> = (0..64)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let back = decode(&encode(&data));
    for (blk, got) in data.chunks(QK).zip(back.chunks(QK)) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = half_scale(scale_byte(amax));
        for (x, y) in blk.iter().zip(got) {
            // the widest gap between codes is 4 steps (8 to 12) of d; values above 12 d clip
            let limit = if x.abs() <= 12.0 * d {
                2.0 * d
            } else {
                x.abs() - 12.0 * d
            };
            assert!((x - y).abs() <= limit + 1e-6, "{x} -> {y} (d={d})");
        }
    }
}

#[test]
fn zeros_tiny_and_huge_values_keep_a_valid_block() {
    assert_eq!(encode(&[0.0; 32]), [vec![0u8], vec![0u8; 16]].concat());
    assert_eq!(decode(&encode(&[0.0; 32])), vec![0.0; 32]);
    // a non-finite amax falls back to scale 0 instead of a NaN scale
    let mut v = [0.5f32; 32];
    v[3] = f32::INFINITY;
    assert_eq!(encode(&v)[0], 0);
    // the largest finite values stay far below the NaN code 255
    assert_eq!(encode(&[3.0e38; 32])[0], 252);
    // the smallest magnitudes use the subnormal scales and stay finite
    assert!(decode(&encode(&[1e-38; 32])).iter().all(|x| x.is_finite()));
}

#[test]
fn a_tie_between_two_codes_takes_the_lower_one() {
    // d = 0.5 (e = 127): 2.5 lies midway between codes 2.0 (index 4) and 3.0 (index 5)
    let mut v = [0.0f32; 32];
    v[0] = 6.0;
    v[1] = 2.5;
    let b = encode(&v);
    assert_eq!(b[1] & 15, 7); // 6.0 -> index 7 (value 12 * 0.5)
    assert_eq!(b[1] >> 4, 0); // element 16 is zero
    assert_eq!(b[2] & 15, 4);
}
