//! Buffer plans: every device buffer the trainer will create, named and sized, so the footprint
//! is checked against the budget before anything is allocated.

use crate::budget::BufferPlan;
use ferrum_ref::gpt2::{Config, Gpt2};

/// Every activation/scratch buffer for a `(batch, t)` shape, named and sized.
pub fn activation_plan(c: &Config, batch: usize, t: usize) -> BufferPlan {
    let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
    let f = |n: usize| (n * 4) as u64;
    let mut p = BufferPlan::default();
    for l in 0..=c.layers {
        p.add(&format!("x{l}"), f(r * d));
    }
    for l in 0..c.layers {
        for (n, len) in [
            ("h1", r * d),
            ("q", r * d),
            ("k", r * d),
            ("v", r * d),
            ("probs", batch * c.heads * t * t),
            ("a", r * d),
            ("x2", r * d),
            ("h2", r * d),
            ("f1", r * h),
            ("act", r * h),
        ] {
            p.add(&format!("blk{l}.{n}"), f(len));
        }
    }
    for (n, len) in [
        ("o", r * d),
        ("f2", r * d),
        ("tmp_c", r * d),
        ("tmp_h", r * h),
        ("hf", r * d),
        ("logits", r * v),
        ("dlogits", r * v),
        ("row_loss", r),
        ("loss", 1),
        ("dhf", r * d),
        ("dx_a", r * d),
        ("dx_b", r * d),
        ("dact", r * h),
        ("df1", r * h),
        ("dh2", r * d),
        ("dx2ln", r * d),
        ("dx2", r * d),
        ("da", r * d),
        ("dq", r * d),
        ("dk", r * d),
        ("dv", r * d),
        ("dh1a", r * d),
        ("dh1b", r * d),
        ("dh1c", r * d),
        ("t1", r * d),
        ("dh1", r * d),
        ("dxln", r * d),
        ("stats", 2 * r),
        ("dot", batch * c.heads * t),
        ("ds", batch * c.heads * t * t),
        ("junk_v", v),
    ] {
        p.add(n, f(len));
    }
    p
}

/// The gradient accumulators micro-batching adds: one buffer per parameter.
pub fn accum_plan(c: &Config) -> BufferPlan {
    let m = Gpt2::random(*c, 1); // shapes only
    let mut p = BufferPlan::default();
    for (i, t) in m.params().iter().enumerate() {
        p.add(&format!("acc{i}"), (t.len() * 4) as u64);
    }
    p
}

/// Buffers that live as long as the trainer: weights, gradients and both Adam moments.
pub fn static_plan(c: &Config) -> BufferPlan {
    let m = Gpt2::random(*c, 1); // shapes only
    let mut p = BufferPlan::default();
    for (i, t) in m.params().iter().enumerate() {
        for kind in ["w", "g", "m", "v"] {
            p.add(&format!("{kind}{i}"), (t.len() * 4) as u64);
        }
    }
    p
}
