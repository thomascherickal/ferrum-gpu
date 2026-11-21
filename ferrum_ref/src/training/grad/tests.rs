use super::*;
use crate::gpt2::Config;
use crate::matmul::rand_vec;

fn tiny() -> Config {
    Config {
        vocab: 7,
        ctx: 5,
        dim: 4,
        heads: 2,
        layers: 2,
        hidden: 6,
    }
}

fn data(cfg: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1) % cfg.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2) % cfg.vocab).collect(),
    )
}

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

#[test]
fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 21);
    let (batch, t) = (2, 4);
    let (ids, targets) = data(&cfg, batch, t);
    let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let h = 1e-2;
    let eval = |m: &Gpt2| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
    let n_params = m.params().len();
    let mut checked = 0;
    for pi in 0..n_params {
        let len = m.params()[pi].len();
        // Check up to 6 spread-out elements of every tensor (all of them when small).
        let picks: Vec<usize> = if len <= 6 {
            (0..len).collect()
        } else {
            (0..6).map(|j| j * (len - 1) / 5).collect()
        };
        for &i in &picks {
            let (mut p, mut q) = (m.clone(), m.clone());
            p.params_mut()[pi][i] += h;
            q.params_mut()[pi][i] -= h;
            let num = (eval(&p) - eval(&q)) / (2.0 * h);
            let ana = g.params()[pi][i];
            assert!(
                (ana - num).abs() < 4e-3 + 2e-2 * num.abs(),
                "param {pi}[{i}]: analytic {ana} vs numeric {num}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} elements checked");
    let norm: f32 = g
        .params()
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    assert!(norm > 1e-3, "gradient is suspiciously zero");
}

#[test]
fn token_embedding_gradient_sums_repeated_tokens() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 22);
    // The same token three times: its row is the sum of three position contributions.
    let ids = vec![2usize, 2, 2, 5];
    let targets = vec![1usize, 3, 4, 0];
    let (_, g) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    assert!(g.tok[2 * 4..3 * 4].iter().any(|v| v.abs() > 1e-6));
    assert!(
        g.tok[..4].iter().all(|&v| v == 0.0),
        "unused token has no gradient"
    );
}

#[test]
fn batching_equals_two_separate_sequences() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 23);
    let (ids, targets) = data(&cfg, 2, 3);
    let (l, g) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l1, g1) = loss_and_grads(&m, &ids[..3], &targets[..3], 1, 1).unwrap();
    let (l2, g2) = loss_and_grads(&m, &ids[3..], &targets[3..], 1, 1).unwrap();
    assert!(
        (l - (l1 + l2) / 2.0).abs() < 1e-5,
        "mean loss is the average of the sequences"
    );
    for ((a, b), c) in g.params().iter().zip(g1.params()).zip(g2.params()) {
        for i in 0..a.len() {
            assert!(
                (a[i] - (b[i] + c[i]) / 2.0).abs() < 1e-5,
                "gradient is the average too"
            );
        }
    }
}

#[test]
fn thread_count_does_not_change_the_result() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 24);
    let (ids, targets) = data(&cfg, 2, 4);
    let one = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    for t in [2, 5] {
        let other = loss_and_grads(&m, &ids, &targets, 2, t).unwrap();
        assert_eq!(one.0, other.0);
        assert_eq!(one.1, other.1, "bit-identical at {t} threads");
    }
}

#[test]
fn input_validation() {
    let m = Gpt2::random(tiny(), 25);
    let e = |ids: &[usize], tg: &[usize], b: usize| loss_and_grads(&m, ids, tg, b, 1).unwrap_err();
    assert!(e(&[], &[], 1).contains("split"));
    assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
    assert!(e(&[1; 6], &[1; 6], 1).contains("exceeds the context"));
    assert!(e(&[1, 2], &[1], 1).contains("targets"));
    assert!(e(&[7], &[1], 1).contains("out of range"));
    assert!(e(&[1], &[9], 1).contains("out of range"));
}

#[test]
fn loss_matches_the_forward_oracle() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 26);
    let (ids, targets) = data(&cfg, 1, 5);
    let logits = m.forward_logits(&ids).unwrap();
    let (want, _) = nn::cross_entropy(&logits, &targets, cfg.vocab);
    let (got, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    assert!((got - want).abs() < 1e-5);
}
