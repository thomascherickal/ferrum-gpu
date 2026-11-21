use super::*;

/// Like [`plan`], but a weight or the embedding larger than `limit` bytes is listed as the blocks
/// it will be stored in (`name.0`, `name.1`, ...), so the binding check sees what is really bound.
/// Buffers that cannot be split (KV cache, scratch) stay whole and fail the check as before.
pub fn plan_limited(c: &Config, quant: Option<QKind>, limit: u64) -> BufferPlan {
    let (d, h, qd, kvd) = (c.dim, c.hidden, c.q_dim(), c.kv_dim());
    let mut p = BufferPlan::default();
    let add = |p: &mut BufferPlan, name: &str, bytes: u64, units: usize| {
        let unit = if units == 0 {
            bytes
        } else {
            bytes / units as u64
        };
        match shard::blocks(units, unit, limit) {
            Ok(b) if b.len() > 1 && shard::splittable(name) => {
                for (i, (_, len)) in b.iter().enumerate() {
                    p.add(&format!("{name}.{i}"), *len as u64 * unit);
                }
            }
            _ => p.add(name, bytes),
        }
    };
    add(&mut p, "tok_emb", f32s(c.vocab * d), c.vocab);
    p.add("rope_cos", f32s(c.ctx * c.rope_dim / 2));
    p.add("rope_sin", f32s(c.ctx * c.rope_dim / 2));
    for i in 0..c.layers {
        for (n, bytes, units) in [
            ("attn_norm", f32s(d), 0),
            ("wq", matrix_bytes(quant, d, qd), qd),
            ("wk", matrix_bytes(quant, d, kvd), kvd),
            ("wv", matrix_bytes(quant, d, kvd), kvd),
            ("wo", matrix_bytes(quant, qd, d), d),
            ("ffn_norm", f32s(d), 0),
            ("gate", matrix_bytes(quant, d, h), h),
            ("up", matrix_bytes(quant, d, h), h),
            ("down", matrix_bytes(quant, h, d), d),
            ("kcache", f32s(c.ctx * kvd), 0),
            ("vcache", f32s(c.ctx * kvd), 0),
        ] {
            add(&mut p, &format!("blk{i}.{n}"), bytes, units);
        }
    }
    p.add("out_norm", f32s(d));
    add(&mut p, "head", matrix_bytes(quant, d, c.vocab), c.vocab);
    for (n, len) in [
        ("zero_q", qd),
        ("zero_kv", kvd),
        ("zero_d", d),
        ("zero_h", h),
        ("zero_vocab", c.vocab),
        ("ids", 1),
        ("x", d),
        ("x2", d),
        ("h", d),
        ("q", qd),
        ("k", kvd),
        ("v", kvd),
        ("attn", qd),
        ("o", d),
        ("gate", h),
        ("up", h),
        ("act", h),
        ("f2", d),
        ("probs", c.heads * c.ctx),
        ("logits", c.vocab),
        ("tmp_q", qd),
        ("tmp_kv", kvd),
    ] {
        p.add(&format!("scratch.{n}"), f32s(len));
    }
    p
}
