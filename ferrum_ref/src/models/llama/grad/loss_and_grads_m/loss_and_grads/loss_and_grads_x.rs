use super::*;

/// Mean next-token cross-entropy over all positions and the gradient of every parameter
/// (returned as a [`Llama`] of the same shape; an absent bias has an empty gradient).
pub fn loss_and_grads(
    m: &Llama,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
) -> Result<(f32, Llama), String> {
    let t = check(m, ids, targets, batch)?;
    let (c, rows) = (&m.cfg, ids.len());
    let (d, hid, hd, qd, kvd) = (c.dim, c.hidden, c.head_dim, c.q_dim(), c.kv_dim());
    let shape = (batch, t, c.heads, c.kv_heads, hd);
    // RoPE tables for positions 0..t, repeated for every sequence of the batch.
    let (c1, s1) = rope_tables(0, t, c.rope_dim, c.rope_base, &m.rope);
    let (cos, sin) = (c1.repeat(batch), s1.repeat(batch));
    let tables = (&cos[..], &sin[..]);
    let mut x: Vec<f32> = ids
        .iter()
        .flat_map(|&i| m.tok[i * d..(i + 1) * d].iter().copied())
        .collect();
    let mut caches = Vec::with_capacity(m.blocks.len());
    for b in &m.blocks {
        let x_in = x.clone();
        let h1 = rmsnorm(&x, &b.attn_norm, rows, d, c.eps);
        let mut q = lin(&h1, &b.wq, &b.bq, (rows, d, qd), threads);
        let mut k = lin(&h1, &b.wk, &b.bk, (rows, d, kvd), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, kvd), threads);
        apply_rope(&mut q, rope_dims(c, rows, c.heads), tables, c.rope_type);
        apply_rope(&mut k, rope_dims(c, rows, c.kv_heads), tables, c.rope_type);
        let (a, probs) = gqa_fwd(&q, &k, &v, shape);
        let x2 = nn::add(&x, &mm(&a, &b.wo, (rows, qd, d), threads));
        let h2 = rmsnorm(&x2, &b.ffn_norm, rows, d, c.eps);
        let gate = mm(&h2, &b.gate, (rows, d, hid), threads);
        let up = mm(&h2, &b.up, (rows, d, hid), threads);
        let act = swiglu(&gate, &up);
        x = nn::add(&x2, &mm(&act, &b.down, (rows, hid, d), threads));
        caches.push(BlockCache {
            x: x_in,
            h1,
            q,
            k,
            v,
            probs,
            a,
            x2,
            h2,
            gate,
            up,
            act,
        });
    }
    let hf = rmsnorm(&x, &m.out_norm, rows, d, c.eps);
    let logits = mm(&hf, &m.head, (rows, d, c.vocab), threads);
    let (loss, dlogits) = nn::cross_entropy(&logits, targets, c.vocab);

    let mut g = m.zeros_like();
    let (dhf, dhead, _) = linear_bwd(&hf, &m.head, &dlogits, (rows, d, c.vocab), threads);
    g.head = dhead;
    let (mut dx, dg) = rmsnorm_bwd(&x, &m.out_norm, &dhf, (rows, d), c.eps);
    g.out_norm = dg;
    for (bi, (b, ca)) in m.blocks.iter().zip(&caches).enumerate().rev() {
        let gb = &mut g.blocks[bi];
        // x_out = x2 + down(act)
        let (dact, ddown, _) = linear_bwd(&ca.act, &b.down, &dx, (rows, hid, d), threads);
        gb.down = ddown;
        let (dgate, dup) = swiglu_bwd(&ca.gate, &ca.up, &dact);
        let (dh2a, dwg, _) = linear_bwd(&ca.h2, &b.gate, &dgate, (rows, d, hid), threads);
        let (dh2b, dwu, _) = linear_bwd(&ca.h2, &b.up, &dup, (rows, d, hid), threads);
        gb.gate = dwg;
        gb.up = dwu;
        let (dx2_n, dgn) = rmsnorm_bwd(
            &ca.x2,
            &b.ffn_norm,
            &nn::add(&dh2a, &dh2b),
            (rows, d),
            c.eps,
        );
        gb.ffn_norm = dgn;
        let dx2 = nn::add(&dx, &dx2_n);
        // x2 = x + wo(a)
        let (da, dwo, _) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, qd, d), threads);
        gb.wo = dwo;
        let (mut dq, mut dk, dv) = gqa_bwd((&ca.q, &ca.k, &ca.v), &ca.probs, &da, shape);
        rope_bwd(&mut dq, rope_dims(c, rows, c.heads), tables, c.rope_type);
        rope_bwd(&mut dk, rope_dims(c, rows, c.kv_heads), tables, c.rope_type);
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, qd), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, kvd), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, kvd), threads);
        gb.wq = dwq;
        gb.wk = dwk;
        gb.wv = dwv;
        for (has, dst, db) in [
            (&b.bq, &mut gb.bq, dbq),
            (&b.bk, &mut gb.bk, dbk),
            (&b.bv, &mut gb.bv, dbv),
        ] {
            if !has.is_empty() {
                *dst = db;
            }
        }
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_n, dg1) = rmsnorm_bwd(&ca.x, &b.attn_norm, &dh1, (rows, d), c.eps);
        gb.attn_norm = dg1;
        dx = nn::add(&dx2, &dx_n);
    }
    // Embedding gradient: rows summed per token id, in row order.
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..d {
            g.tok[id * d + i] += dx[r * d + i];
        }
    }
    Ok((loss, g))
}
