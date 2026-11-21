use super::*;

pub(super) fn f(n: usize) -> u64 {
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
