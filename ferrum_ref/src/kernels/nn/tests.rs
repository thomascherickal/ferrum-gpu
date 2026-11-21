use super::*;
use crate::matmul::rand_vec;

#[test]
fn gelu_known_values() {
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
    assert!((gelu(-1.0) + 0.158_808).abs() < 1e-5);
    assert_eq!(gelu(20.0), 20.0);
}

#[test]
fn bias_and_add() {
    assert_eq!(
        add_bias(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0]),
        vec![11.0, 22.0, 13.0, 24.0]
    );
    assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
}

#[test]
fn layernorm_normalises_each_row() {
    let x = rand_vec(3 * 8, 1);
    let y = layernorm(&x, &[1.0; 8], &[0.0; 8], 3, 8);
    for r in 0..3 {
        let row = &y[r * 8..(r + 1) * 8];
        let mean: f32 = row.iter().sum::<f32>() / 8.0;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3);
    }
    // gain and bias apply after normalising
    let y2 = layernorm(&x, &[2.0; 8], &[1.0; 8], 3, 8);
    assert!((y2[0] - (2.0 * y[0] + 1.0)).abs() < 1e-5);
}

#[test]
fn attention_is_causal_and_rows_sum_to_one() {
    let (batch, t, heads, hd) = (2, 5, 2, 3);
    let n = batch * t * heads * hd;
    let (q, k, v) = (rand_vec(n, 1), rand_vec(n, 2), rand_vec(n, 3));
    let (out, probs) = attention(&q, &k, &v, (batch, t, heads, hd));
    assert_eq!(out.len(), n);
    for row in 0..batch * heads * t {
        let p = &probs[row * t..(row + 1) * t];
        let i = row % t;
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(
            p[i + 1..].iter().all(|&x| x == 0.0),
            "future positions must be masked"
        );
    }
    // the first token attends only to itself, so out == v there
    assert!((out[0] - v[0]).abs() < 1e-6);
}

#[test]
fn a_nan_model_gives_a_nan_loss_instead_of_a_clamped_finite_one() {
    // clamp_prob lets NaN through on purpose: f32::max(NaN, x) would hide divergence.
    let (loss, _) = cross_entropy(&[f32::NAN, 0.0, 1.0], &[0], 3);
    assert!(
        loss.is_nan(),
        "a NaN probability must not be clamped to a finite loss"
    );
    // A zero probability, by contrast, is clamped so the loss stays finite.
    let (loss, _) = cross_entropy(&[-1e30, 0.0], &[0], 2);
    assert!(loss.is_finite() && loss > 20.0);
}

#[test]
fn cross_entropy_known_loss_and_gradient_sums_to_zero() {
    let (loss, g) = cross_entropy(&[0.0, 0.0, 0.0, 0.0], &[1], 4);
    assert!((loss - 4f32.ln()).abs() < 1e-6);
    assert!(g.iter().sum::<f32>().abs() < 1e-6);
    assert!((g[1] - (0.25 - 1.0)).abs() < 1e-6);
    let (_, g2) = cross_entropy(&rand_vec(2 * 6, 4), &[0, 5], 6);
    assert!(g2.iter().sum::<f32>().abs() < 1e-5);
}

#[test]
fn score_logprob_and_argmax_flag() {
    let s = score(&[1.0, 3.0, 2.0, 5.0, 5.0, 1.0], &[1, 1], 3);
    assert!(s[0].1, "target 1 is the argmax of row 0");
    assert!(
        !s[1].1,
        "ties resolve to the first maximum, which is index 0"
    );
    let (loss, _) = cross_entropy(&[1.0, 3.0, 2.0], &[1], 3);
    assert!((s[0].0 + loss).abs() < 1e-5);
}

#[test]
fn embed_adds_token_and_position_rows() {
    let tok = [1.0, 2.0, 10.0, 20.0]; // vocab 2, c 2
    let pos = [0.1, 0.2, 0.3, 0.4]; // t 2
    assert_eq!(
        embed(&tok, &pos, &[1, 0, 1], 2, 2),
        vec![10.1, 20.2, 1.3, 2.4, 10.1, 20.2]
    );
}
