use super::*;

#[test]
fn e4m3_codes_have_the_published_values() {
    let f = Fp8::E4M3;
    assert_eq!(f.decode(0x00), 0.0);
    assert_eq!(f.decode(0x01), 2f32.powi(-9)); // the smallest subnormal
    assert_eq!(f.decode(0x08), 2f32.powi(-6)); // the smallest normal
    assert_eq!(f.decode(0x38), 1.0);
    assert_eq!(f.decode(0x7e), 448.0); // the largest finite value
    assert!(f.decode(0x7f).is_nan() && f.decode(0xff).is_nan());
    assert_eq!(f.decode(0xb8), -1.0);
}

#[test]
fn e5m2_codes_have_the_published_values() {
    let f = Fp8::E5M2;
    assert_eq!(f.decode(0x01), 2f32.powi(-16));
    assert_eq!(f.decode(0x04), 2f32.powi(-14));
    assert_eq!(f.decode(0x3c), 1.0);
    assert_eq!(f.decode(0x7b), 57344.0);
    assert_eq!(f.decode(0x7c), f32::INFINITY);
    assert_eq!(f.decode(0xfc), f32::NEG_INFINITY);
    assert!(f.decode(0x7d).is_nan());
}

#[test]
fn every_finite_code_round_trips_through_encode() {
    for f in [Fp8::E4M3, Fp8::E5M2] {
        let limit = if f == Fp8::E4M3 { 0x7e } else { 0x7b };
        for code in 0..=limit {
            let v = f.decode(code);
            assert_eq!(f.decode(f.encode(v)), v, "{f:?} {code:#x}");
            assert_eq!(f.decode(f.encode(-v)), -v, "{f:?} -{code:#x}");
        }
    }
}

#[test]
fn encode_rounds_to_the_nearest_code_ties_to_even_and_saturates() {
    let f = Fp8::E4M3;
    // between 1.0 (0x38) and 1.125 (0x39): the midpoint goes to the even code
    assert_eq!(f.encode(1.0625), 0x38);
    assert_eq!(f.encode(1.0626), 0x39);
    // between 1.125 (0x39) and 1.25 (0x3a)
    assert_eq!(f.encode(1.1875), 0x3a);
    // saturation and sign of zero
    assert_eq!(f.encode(1.0e9), 0x7e);
    assert_eq!(f.encode(-1.0e9), 0xfe);
    assert_eq!(f.encode(f32::INFINITY), 0x7e);
    assert_eq!(f.encode(-0.0), 0x80);
    assert_eq!(f.encode(f32::NAN), 0x7f);
    assert_eq!(Fp8::E5M2.encode(f32::NAN), 0x7e);
    assert_eq!(Fp8::E5M2.encode(1.0e9), 0x7b);
}

#[test]
fn blocks_scale_to_the_formats_range_and_keep_the_relative_error_small() {
    let data: Vec<f32> = (0..64)
        .map(|i| ((i as f32) * 0.31).sin() * (1.0 + (i % 5) as f32))
        .collect();
    for (f, tol) in [(Fp8::E4M3, 0.07), (Fp8::E5M2, 0.13)] {
        let bytes = encode(f, &data);
        assert_eq!(bytes.len(), 2 * BLOCK);
        let back = decode(f, &bytes);
        for (blk, got) in data.chunks(32).zip(back.chunks(32)) {
            let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            for (x, y) in blk.iter().zip(got) {
                assert!((x - y).abs() <= tol * amax, "{f:?} {x} -> {y}");
            }
        }
    }
    // an all-zero block has a zero scale and decodes to zero
    assert_eq!(
        decode(Fp8::E4M3, &encode(Fp8::E4M3, &[0.0; 32])),
        vec![0.0; 32]
    );
}
