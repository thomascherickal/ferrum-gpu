#![allow(clippy::needless_range_loop)] // index loops read clearer in element-wise finite-difference checks

use super::grad::*;
use super::ops::{apply_rope, attend_row, rmsnorm, rope_tables, swiglu, RopeScaling, RopeType};
use super::*;
use crate::matmul::rand_vec;
use crate::optim::{Adam, Schedule};
use crate::trainer::StepConfig;

fn tiny(ty: RopeType) -> Config {
    Config {
        vocab: 7,
        ctx: 5,
        dim: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 2,
        layers: 2,
        hidden: 6,
        rope_dim: 2,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

fn data(cfg: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    let n = batch * t;
    (
        (0..n).map(|i| (i * 3 + 1) % cfg.vocab).collect(),
        (0..n).map(|i| (i * 5 + 2) % cfg.vocab).collect(),
    )
}

/// Central differences of `f` at every index of `x` against the analytic gradient `g`.
fn check_fd(name: &str, x: &[f32], g: &[f32], f: &dyn Fn(&[f32]) -> f32) {
    let h = 1e-2;
    for i in 0..x.len() {
        let (mut p, mut q) = (x.to_vec(), x.to_vec());
        p[i] += h;
        q[i] -= h;
        let num = (f(&p) - f(&q)) / (2.0 * h);
        assert!(
            (g[i] - num).abs() < 2e-3 + 1e-2 * num.abs(),
            "{name}[{i}]: analytic {} vs numeric {num}",
            g[i]
        );
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn rmsnorm_bwd_matches_finite_differences() {
    let (rows, cols) = (3, 5);
    let (x, gain, dy) = (
        rand_vec(rows * cols, 1),
        rand_vec(cols, 2),
        rand_vec(rows * cols, 3),
    );
    let (dx, dg) = rmsnorm_bwd(&x, &gain, &dy, (rows, cols), 1e-5);
    check_fd("dx", &x, &dx, &|x| {
        dot(&rmsnorm(x, &gain, rows, cols, 1e-5), &dy)
    });
    check_fd("dgain", &gain, &dg, &|g| {
        dot(&rmsnorm(&x, g, rows, cols, 1e-5), &dy)
    });
}

#[test]
fn swiglu_bwd_matches_finite_differences_and_saturates() {
    let (gate, up, dact) = (rand_vec(12, 4), rand_vec(12, 5), rand_vec(12, 6));
    let (dg, du) = swiglu_bwd(&gate, &up, &dact);
    check_fd("dgate", &gate, &dg, &|g| dot(&swiglu(g, &up), &dact));
    check_fd("dup", &up, &du, &|u| dot(&swiglu(&gate, u), &dact));
    let (dg, du) = swiglu_bwd(&[-200.0, 200.0, 0.0], &[1.0; 3], &[1.0; 3]);
    assert!(
        dg.iter().chain(&du).all(|v| v.is_finite()),
        "saturation must not give NaN"
    );
    assert_eq!(dg[0], 0.0);
    assert!((dg[1] - 1.0).abs() < 1e-6 && (dg[2] - 0.5).abs() < 1e-6);
}

#[test]
fn rope_bwd_is_the_transpose_of_rope_for_both_pairings() {
    for ty in [RopeType::Norm, RopeType::Neox] {
        let (rows, heads, hd, rd) = (3, 2, 6, 4);
        let sc = RopeScaling {
            freq_factors: vec![1.0, 3.0],
            linear_factor: 1.5,
        };
        let (c, s) = rope_tables(2, rows, rd, 10000.0, &sc);
        let (x, y) = (
            rand_vec(rows * heads * hd, 7),
            rand_vec(rows * heads * hd, 8),
        );
        let mut rx = x.clone();
        apply_rope(&mut rx, (rows, heads, hd, rd), (&c, &s), ty);
        let mut ty_ = y.clone();
        rope_bwd(&mut ty_, (rows, heads, hd, rd), (&c, &s), ty);
        // <R x, y> == <x, R^T y>
        assert!((dot(&rx, &y) - dot(&x, &ty_)).abs() < 1e-4, "{ty:?}");
        // and R^T R = I
        let mut back = rx.clone();
        rope_bwd(&mut back, (rows, heads, hd, rd), (&c, &s), ty);
        assert!(
            back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5),
            "{ty:?}"
        );
    }
}

#[test]
fn gqa_forward_agrees_with_the_row_oracle() {
    let shape = (2, 4, 4, 2, 3);
    let (batch, t, heads, kv, hd) = shape;
    let (q, k, v) = (
        rand_vec(batch * t * heads * hd, 1),
        rand_vec(batch * t * kv * hd, 2),
        rand_vec(batch * t * kv * hd, 3),
    );
    let (out, probs) = gqa_fwd(&q, &k, &v, shape);
    let (qd, kvd) = (heads * hd, kv * hd);
    for b in 0..batch {
        for i in 0..t {
            let r = b * t + i;
            let want = attend_row(
                &q[r * qd..(r + 1) * qd],
                &k[b * t * kvd..],
                &v[b * t * kvd..],
                i + 1,
                (heads, kv, hd),
            );
            assert!(out[r * qd..(r + 1) * qd]
                .iter()
                .zip(&want)
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }
    for row in 0..batch * heads * t {
        let p = &probs[row * t..(row + 1) * t];
        assert!(
            (p.iter().sum::<f32>() - 1.0).abs() < 1e-5
                && p[row % t + 1..].iter().all(|&x| x == 0.0)
        );
    }
}

#[test]
fn gqa_bwd_matches_finite_differences_for_every_group_ratio() {
    for (heads, kv) in [(4usize, 4usize), (4, 2), (4, 1)] {
        let shape = (2, 3, heads, kv, 2);
        let (batch, t, _, _, hd) = shape;
        let (q, k, v) = (
            rand_vec(batch * t * heads * hd, 1),
            rand_vec(batch * t * kv * hd, 2),
            rand_vec(batch * t * kv * hd, 3),
        );
        let dout = rand_vec(batch * t * heads * hd, 4);
        let (_, probs) = gqa_fwd(&q, &k, &v, shape);
        let (dq, dk, dv) = gqa_bwd((&q, &k, &v), &probs, &dout, shape);
        let f = |q: &[f32], k: &[f32], v: &[f32]| dot(&gqa_fwd(q, k, v, shape).0, &dout);
        let tag = format!("{heads}/{kv}");
        check_fd(&format!("{tag} dq"), &q, &dq, &|x| f(x, &k, &v));
        check_fd(&format!("{tag} dk"), &k, &dk, &|x| f(&q, x, &v));
        check_fd(&format!("{tag} dv"), &v, &dv, &|x| f(&q, &k, x));
    }
}

fn variants() -> Vec<(&'static str, Llama)> {
    let mut mqa = tiny(RopeType::Norm);
    mqa.kv_heads = 1;
    mqa.head_dim = 4;
    mqa.rope_dim = 2; // partial rotary
    let mut scaled = Llama::random(tiny(RopeType::Norm), 25, false, false);
    scaled.rope = RopeScaling {
        freq_factors: vec![2.0],
        linear_factor: 2.0,
    };
    vec![
        (
            "norm",
            Llama::random(tiny(RopeType::Norm), 21, false, false),
        ),
        (
            "neox+bias",
            Llama::random(tiny(RopeType::Neox), 22, true, false),
        ),
        ("mqa+partial", Llama::random(mqa, 23, true, false)),
        ("scaled", scaled),
    ]
}

#[test]
fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    for (name, m) in variants() {
        let (batch, t) = (2, 4);
        let (ids, targets) = data(&m.cfg, batch, t);
        let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
        let eval = |m: &Llama| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
        let h = 1e-2;
        let mut checked = 0;
        for pi in 0..m.params().len() {
            let len = m.params()[pi].len();
            if len == 0 {
                assert!(
                    g.params()[pi].is_empty(),
                    "{name}: an absent bias has no gradient"
                );
                continue;
            }
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
                    "{name} param {pi}[{i}]: analytic {ana} vs numeric {num}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "{name}: only {checked} elements checked");
    }
}

#[test]
fn token_embedding_gradient_sums_repeated_tokens() {
    let m = Llama::random(tiny(RopeType::Norm), 26, false, false);
    let ids = vec![2usize, 2, 2, 5];
    let (_, g) = loss_and_grads(&m, &ids, &[1, 3, 4, 0], 1, 1).unwrap();
    let d = m.cfg.dim;
    assert!(g.tok[2 * d..3 * d].iter().any(|&v| v != 0.0));
    assert!(
        g.tok[..d].iter().all(|&v| v == 0.0),
        "an unused token has no gradient"
    );
}

#[test]
fn batching_equals_two_separate_sequences_and_threads_do_not_matter() {
    let m = Llama::random(tiny(RopeType::Neox), 27, true, false);
    let (ids, targets) = data(&m.cfg, 2, 3);
    let (l, g) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l0, g0) = loss_and_grads(&m, &ids[..3], &targets[..3], 1, 1).unwrap();
    let (l1, g1) = loss_and_grads(&m, &ids[3..], &targets[3..], 1, 1).unwrap();
    assert!((l - (l0 + l1) / 2.0).abs() < 1e-5);
    for (pi, p) in g.params().iter().enumerate() {
        for (j, &v) in p.iter().enumerate() {
            let want = (g0.params()[pi][j] + g1.params()[pi][j]) / 2.0;
            assert!((v - want).abs() < 1e-5, "param {pi}[{j}]");
        }
    }
    let (l4, g4) = loss_and_grads(&m, &ids, &targets, 2, 4).unwrap();
    assert!((l - l4).abs() < 1e-6);
    assert!(g
        .params()
        .iter()
        .zip(g4.params())
        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)));
}

#[test]
fn loss_matches_the_forward_oracle_and_bad_input_is_refused() {
    let m = Llama::random(tiny(RopeType::Norm), 28, false, false);
    let (ids, targets) = data(&m.cfg, 1, 4);
    let (loss, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    let want = nn::cross_entropy(&m.forward_logits(&ids).unwrap(), &targets, 7).0;
    assert!((loss - want).abs() < 1e-5, "{loss} vs {want}");
    let e = |r: Result<(f32, Llama), String>| r.expect_err("must refuse");
    assert!(e(loss_and_grads(&m, &ids, &targets, 3, 1)).contains("split"));
    assert!(e(loss_and_grads(&m, &[0; 6], &[0; 6], 1, 1)).contains("context"));
    assert!(e(loss_and_grads(&m, &ids, &targets[..3], 1, 1)).contains("targets"));
    assert!(e(loss_and_grads(&m, &[7, 0, 0, 0], &targets, 1, 1)).contains("out of range"));
    let tied = Llama::random(tiny(RopeType::Norm), 29, false, true);
    assert!(e(loss_and_grads(&tied, &ids, &targets, 1, 1)).contains("tied"));
}

#[test]
fn param_accessors_agree_with_each_other_and_with_decay_rules() {
    let m = Llama::random(tiny(RopeType::Neox), 30, true, false);
    let n = m.params().len();
    assert_eq!(n, 1 + 2 * BLOCK_PARAMS + 2);
    assert_eq!(
        m.params().iter().map(|p| p.len()).sum::<usize>(),
        m.num_params()
    );
    let z = m.zeros_like();
    assert!(z.params().iter().all(|p| p.iter().all(|&v| v == 0.0)));
    assert_eq!(
        z.params().iter().map(|p| p.len()).collect::<Vec<_>>(),
        m.params().iter().map(|p| p.len()).collect::<Vec<_>>()
    );
    let decayed: Vec<bool> = (0..n).map(|i| m.decays(i)).collect();
    assert!(decayed[0] && decayed[n - 1], "embedding and head decay");
    assert!(!decayed[n - 2], "output norm does not");
    // per block: norms and biases do not decay, matrices do
    for blk in 0..2 {
        let o = 1 + blk * BLOCK_PARAMS;
        let want = [
            false, true, false, true, false, true, false, true, false, true, true, true,
        ];
        assert_eq!(&decayed[o..o + BLOCK_PARAMS], &want);
    }
    // every decayed parameter is a matrix (more than a vector's worth of elements per row)
    assert_eq!(m.params()[1 + 1].len(), 8 * 8); // wq of block 0
}

#[test]
fn adam_step_matches_the_hand_computed_first_update_and_clipping_caps_the_norm() {
    let mut m = Llama::random(tiny(RopeType::Norm), 31, false, false);
    let mut g = m.zeros_like();
    g.out_norm.fill(2.0);
    g.head[0] = -3.0;
    let before = m.clone();
    let mut st = Moments::new(&m);
    let adam = Adam {
        weight_decay: 0.1,
        ..Adam::new(0.01)
    };
    adam_step(&mut m, &g, &mut st, &adam, 1.0);
    // step 1: m_hat = g, v_hat = g^2, so the update is lr * g / (|g| + eps) = lr * sign(g)
    assert!(
        (m.out_norm[0] - (before.out_norm[0] - 0.01)).abs() < 1e-6,
        "no decay on the norm gain"
    );
    let want = before.head[0] + 0.01 - 0.01 * 0.1 * before.head[0];
    assert!(
        (m.head[0] - want).abs() < 1e-6,
        "decayed head: {} vs {want}",
        m.head[0]
    );
    // zero gradient: only the decoupled decay moves a decayed tensor, by lr * wd * p
    for (a, b) in m.tok.iter().zip(&before.tok) {
        assert!((a - (b - 0.01 * 0.1 * b)).abs() < 1e-7);
    }
    assert_eq!(
        m.blocks[0].attn_norm, before.blocks[0].attn_norm,
        "norm gains never decay"
    );
    assert_eq!(st.t, 1);
    assert!((global_norm(&g) - (8.0f32 * 4.0 + 9.0).sqrt()).abs() < 1e-5);
}

#[test]
fn training_a_fixed_batch_drives_the_loss_down() {
    let m = Llama::random(tiny(RopeType::Norm), 32, false, false);
    let (ids, targets) = data(&m.cfg, 2, 4);
    let mut tr = LlamaTrainer::new(m, 1);
    let sched = Schedule {
        base_lr: 0.05,
        warmup: 2,
        total: 40,
    };
    let mut losses = Vec::new();
    for step in 1..=40u64 {
        let cfg = StepConfig {
            adam: Adam::new(sched.lr_at(step)),
            clip: 1.0,
        };
        losses.push(tr.step(&ids, &targets, 2, &cfg).unwrap().loss);
    }
    assert_eq!(tr.steps(), 40);
    assert!(
        losses[39] < 0.3 * losses[0],
        "loss {} -> {}",
        losses[0],
        losses[39]
    );
    assert!(losses.iter().all(|l| l.is_finite()));
}
