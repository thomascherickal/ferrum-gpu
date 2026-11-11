//! Buffer plans of the Llama trainer: every device buffer, named and sized, so the footprint is
//! checked against the budget before anything is allocated.

use crate::budget::BufferPlan;
use crate::ops_opt::parts_for;
use ferrum_ref::llama::{Config, Llama};

fn f(n: usize) -> u64 {
    (n * 4) as u64
}

/// Element counts of every trainable tensor in canonical order, for a model of shape `c`
/// (`biases`: whether q, k and v carry a bias, as in Qwen2). Biases that are absent are empty.
pub fn param_sizes(c: &Config, biases: bool) -> Vec<usize> {
    let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
    let b = |n: usize| if biases { n } else { 0 };
    let mut v = vec![c.vocab * d];
    for _ in 0..c.layers {
        v.extend([d, d * qd, b(qd), d * kvd, b(kvd), d * kvd, b(kvd), qd * d]);
        v.extend([d, d * h, d * h, h * d]);
    }
    v.extend([d, d * c.vocab]);
    v
}

/// Weights, gradients, both Adam moments and the norm partials for tensors of these sizes: what
/// lives for the whole run.
pub fn static_plan_for(sizes: &[usize]) -> BufferPlan {
    let mut p = BufferPlan::default();
    let mut parts = 0;
    for (i, &n) in sizes.iter().enumerate() {
        for set in ["param", "grad", "m", "v"] {
            p.add(&format!("{set}{i}"), f(n));
        }
        parts += parts_for(n);
    }
    p.add("norm_partials", f(parts));
    p.add("scale", f(2));
    p
}

/// [`static_plan_for`] the tensors of `m`.
pub fn static_plan(m: &Llama) -> BufferPlan {
    let sizes: Vec<usize> = m.params().iter().map(|p| p.len()).collect();
    static_plan_for(&sizes)
}

/// Every activation and scratch buffer for a `(batch, t)` shape.
pub fn activation_plan(c: &Config, batch: usize, t: usize) -> BufferPlan {
    let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
    let (qd, kvd, half) = (c.q_dim(), c.kv_dim(), c.rope_dim / 2);
    let mut p = BufferPlan::default();
    for l in 0..=c.layers {
        p.add(&format!("x{l}"), f(r * d));
    }
    for l in 0..c.layers {
        for (n, len) in [
            ("h1", r * d),
            ("q", r * qd),
            ("k", r * kvd),
            ("v", r * kvd),
            ("probs", batch * c.heads * t * t),
            ("a", r * qd),
            ("x2", r * d),
            ("h2", r * d),
            ("gate", r * h),
            ("up", r * h),
            ("act", r * h),
        ] {
            p.add(&format!("blk{l}.{n}"), f(len));
        }
    }
    for (n, len) in [
        ("cos", r * half),
        ("sin", r * half),
        ("neg_sin", r * half),
        ("o", r * d),
        ("f2", r * d),
        ("tmp_q", r * qd),
        ("tmp_kv", r * kvd),
        ("hf", r * d),
        ("logits", r * v),
        ("dlogits", r * v),
        ("row_loss", r),
        ("loss", 1),
        ("dhf", r * d),
        ("dx_a", r * d),
        ("dx_b", r * d),
        ("dact", r * h),
        ("dgate", r * h),
        ("dup", r * h),
        ("dh2a", r * d),
        ("dh2b", r * d),
        ("dh2", r * d),
        ("dx2n", r * d),
        ("dx2", r * d),
        ("da", r * qd),
        ("dq", r * qd),
        ("dk", r * kvd),
        ("dv", r * kvd),
        ("dh1a", r * d),
        ("dh1b", r * d),
        ("dh1c", r * d),
        ("t1", r * d),
        ("dh1", r * d),
        ("dxn", r * d),
        ("stats", r),
        ("dot", batch * c.heads * t),
        ("ds", batch * c.heads * t * t),
        ("junk_pos", c.ctx * d),
    ] {
        p.add(n, f(len));
    }
    p
}
