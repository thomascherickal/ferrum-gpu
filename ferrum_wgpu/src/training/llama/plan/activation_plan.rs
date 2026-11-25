use super::*;

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
    ]
    .into_iter()
    .chain(crate::training::HeadActs::rows(r, d, v))
    .chain([
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
    ]) {
        p.add(n, f(len));
    }
    p
}
