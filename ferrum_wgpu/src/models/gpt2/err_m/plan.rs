use super::*;

/// Every device buffer a model of this shape needs, named and sized (weights, KV cache, scratch).
pub fn plan(c: &Config) -> BufferPlan {
    let (d, h) = (c.dim, c.hidden);
    let mut p = BufferPlan::default();
    p.add("tok_emb", f32s(c.vocab * d));
    p.add("pos_emb", f32s(c.ctx * d));
    for i in 0..c.layers {
        for (n, len) in [
            ("ln1_g", d),
            ("ln1_b", d),
            ("wq", d * d),
            ("bq", d),
            ("wk", d * d),
            ("bk", d),
            ("wv", d * d),
            ("bv", d),
            ("wo", d * d),
            ("bo", d),
            ("ln2_g", d),
            ("ln2_b", d),
            ("w1", d * h),
            ("b1", h),
            ("w2", h * d),
            ("b2", d),
            ("kcache", c.ctx * d),
            ("vcache", c.ctx * d),
        ] {
            p.add(&format!("blk{i}.{n}"), f32s(len));
        }
    }
    p.add("lnf_g", f32s(d));
    p.add("lnf_b", f32s(d));
    p.add("head", f32s(d * c.vocab));
    p.add("zero_vocab", f32s(c.vocab));
    for (n, len) in [
        ("ids", 1),
        ("x", d),
        ("x2", d),
        ("h", d),
        ("q", d),
        ("k", d),
        ("v", d),
        ("attn", d),
        ("o", d),
        ("f1", h),
        ("act", h),
        ("f2", d),
        ("probs", c.heads * c.ctx),
        ("logits", c.vocab),
    ] {
        p.add(&format!("scratch.{n}"), f32s(len));
    }
    p
}
