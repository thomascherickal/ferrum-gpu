use super::*;

#[test]
fn linear_bwd_matches_finite_differences() {
    let (m, k, n) = (3, 4, 5);
    let (x, w, dy) = (rand_vec(m * k, 1), rand_vec(k * n, 2), rand_vec(m * n, 3));
    let (dx, dw, db) = linear_bwd(&x, &w, &dy, (m, k, n), 1);
    let b = rand_vec(n, 4);
    let loss = |x: &[f32], w: &[f32], b: &[f32]| -> f32 {
        nn::add_bias(&mm(x, w, (m, k, n), 1), b)
            .iter()
            .zip(&dy)
            .map(|(a, d)| a * d)
            .sum()
    };
    let h = 1e-2;
    for i in 0..x.len() {
        let (mut p, mut q) = (x.clone(), x.clone());
        p[i] += h;
        q[i] -= h;
        assert!(
            (dx[i] - (loss(&p, &w, &b) - loss(&q, &w, &b)) / (2.0 * h)).abs() < 1e-3,
            "dx[{i}]"
        );
    }
    for i in 0..w.len() {
        let (mut p, mut q) = (w.clone(), w.clone());
        p[i] += h;
        q[i] -= h;
        assert!(
            (dw[i] - (loss(&x, &p, &b) - loss(&x, &q, &b)) / (2.0 * h)).abs() < 1e-3,
            "dw[{i}]"
        );
    }
    for i in 0..n {
        let (mut p, mut q) = (b.clone(), b.clone());
        p[i] += h;
        q[i] -= h;
        assert!(
            (db[i] - (loss(&x, &w, &p) - loss(&x, &w, &q)) / (2.0 * h)).abs() < 1e-3,
            "db[{i}]"
        );
    }
}

#[test]
fn layernorm_bwd_matches_finite_differences() {
    let (rows, cols) = (3, 6);
    let (x, g, b, dy) = (
        rand_vec(rows * cols, 5),
        rand_vec(cols, 6),
        rand_vec(cols, 7),
        rand_vec(rows * cols, 8),
    );
    let (dx, dg, db) = layernorm_bwd(&x, &g, &dy, rows, cols);
    let loss = |x: &[f32], g: &[f32], b: &[f32]| -> f32 {
        nn::layernorm(x, g, b, rows, cols)
            .iter()
            .zip(&dy)
            .map(|(a, d)| a * d)
            .sum()
    };
    let h = 1e-2;
    let fd = |f: &dyn Fn(&[f32]) -> f32, v: &[f32], i: usize| {
        let (mut p, mut q) = (v.to_vec(), v.to_vec());
        p[i] += h;
        q[i] -= h;
        (f(&p) - f(&q)) / (2.0 * h)
    };
    for i in 0..x.len() {
        assert!(
            (dx[i] - fd(&|v| loss(v, &g, &b), &x, i)).abs() < 2e-3,
            "dx[{i}]"
        );
    }
    for i in 0..cols {
        assert!(
            (dg[i] - fd(&|v| loss(&x, v, &b), &g, i)).abs() < 2e-3,
            "dg[{i}]"
        );
        assert!(
            (db[i] - fd(&|v| loss(&x, &g, v), &b, i)).abs() < 2e-3,
            "db[{i}]"
        );
    }
}

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
