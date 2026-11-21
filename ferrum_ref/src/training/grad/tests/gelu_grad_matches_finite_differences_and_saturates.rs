use super::*;

#[test]
fn gelu_grad_matches_finite_differences_and_saturates() {
    for x in [-9.0f32, -3.0, -1.0, -0.2, 0.0, 0.5, 2.0, 7.0] {
        let h = 1e-3;
        let num = (nn::gelu(x + h) - nn::gelu(x - h)) / (2.0 * h);
        assert!((gelu_grad(x) - num).abs() < 2e-3, "gelu'({x})");
    }
    assert_eq!(gelu_grad(10.0), 1.0);
    assert_eq!(gelu_grad(-10.0), 0.0);
    assert_eq!(gelu_grad(1e9), 1.0);
    assert_eq!(gelu_grad(0.0), 0.5);
}

#[test]
fn attention_bwd_matches_finite_differences() {
    let shape = (2usize, 4usize, 2usize, 3usize);
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let (q, k, v, dout) = (
        rand_vec(n, 11),
        rand_vec(n, 12),
        rand_vec(n, 13),
        rand_vec(n, 14),
    );
    let (_, probs) = nn::attention(&q, &k, &v, shape);
    let (dq, dk, dv) = attention_bwd(&q, &k, &v, &probs, &dout, shape);
    let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
        nn::attention(q, k, v, shape)
            .0
            .iter()
            .zip(&dout)
            .map(|(a, d)| a * d)
            .sum()
    };
    let h = 1e-2;
    for i in 0..n {
        let step = |v: &[f32]| {
            let (mut p, mut q2) = (v.to_vec(), v.to_vec());
            p[i] += h;
            q2[i] -= h;
            (p, q2)
        };
        let (p, m) = step(&q);
        assert!(
            (dq[i] - (loss(&p, &k, &v) - loss(&m, &k, &v)) / (2.0 * h)).abs() < 2e-3,
            "dq[{i}]"
        );
        let (p, m) = step(&k);
        assert!(
            (dk[i] - (loss(&q, &p, &v) - loss(&q, &m, &v)) / (2.0 * h)).abs() < 2e-3,
            "dk[{i}]"
        );
        let (p, m) = step(&v);
        assert!(
            (dv[i] - (loss(&q, &k, &p) - loss(&q, &k, &m)) / (2.0 * h)).abs() < 2e-3,
            "dv[{i}]"
        );
    }
}
