use super::*;
use crate::blocks::dequant;
use crate::GGML_Q2_K;

fn wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8 + ((i * 7 % 13) as f32 - 6.0) * 0.02)
        .collect()
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
fn nearest_int_rounds_half_to_even_like_ggml() {
    for (x, want) in [
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (-0.5, 0),
        (-1.5, -2),
        (3.7, 4),
        (-3.7, -4),
        (0.0, 0),
    ] {
        assert_eq!(nearest_int(x), want, "{x}");
    }
}

#[test]
fn a_block_is_84_bytes_and_decodes_to_256_values() {
    let data = wave(512);
    let bytes = encode(&data);
    assert_eq!(bytes.len(), 2 * BLOCK);
    assert_eq!(BLOCK, 84);
    assert_eq!(dequant(GGML_Q2_K, &bytes).unwrap().len(), 512);
}

#[test]
fn the_error_is_far_below_the_signal_and_every_sub_block_is_close() {
    let data = wave(1024);
    let back = dequant(GGML_Q2_K, &encode(&data)).unwrap();
    let signal = (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt();
    assert!(
        rmse(&data, &back) < 0.30 * signal,
        "rmse {} vs signal {signal}",
        rmse(&data, &back)
    );
    // two bits per value: each value is within one quantisation step (a third of the range)
    for (sub, got) in data.chunks(16).zip(back.chunks(16)) {
        let range = sub.iter().cloned().fold(f32::MIN, f32::max)
            - sub.iter().cloned().fold(f32::MAX, f32::min);
        for (x, y) in sub.iter().zip(got) {
            assert!(
                (x - y).abs() <= range / 3.0 + 0.02,
                "{x} -> {y} (range {range})"
            );
        }
    }
}

#[test]
fn constant_zero_and_all_negative_blocks_are_valid() {
    // all zero: scale and min are zero, every quant is zero, the decode is zero
    let z = encode(&[0.0; 256]);
    assert!(z[..80].iter().all(|&b| b == 0) && z[80..] == [0, 0, 0, 0]);
    assert!(dequant(GGML_Q2_K, &z).unwrap().iter().all(|&v| v == 0.0));
    // a constant block of positive values has min clamped to zero and decodes close to it
    let c = dequant(GGML_Q2_K, &encode(&[0.5; 256])).unwrap();
    assert!(c.iter().all(|&v| (v - 0.5).abs() < 0.2), "{:?}", &c[..4]);
    // all negative values use the min term
    let n = dequant(GGML_Q2_K, &encode(&[-0.5; 256])).unwrap();
    assert!(n.iter().all(|&v| (v + 0.5).abs() < 0.05), "{:?}", &n[..4]);
}

#[test]
fn encoding_is_deterministic_and_each_block_is_independent() {
    let data = wave(512);
    let a = encode(&data);
    assert_eq!(a, encode(&data));
    assert_eq!(a[..BLOCK], encode(&data[..256])[..]);
    assert_eq!(a[BLOCK..], encode(&data[256..])[..]);
}
