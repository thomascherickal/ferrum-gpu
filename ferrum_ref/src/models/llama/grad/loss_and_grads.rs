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
    let d = c.dim;
    // RoPE tables for positions 0..t, repeated for every sequence of the batch.
    let (c1, s1) = rope_tables(0, t, c.rope_dim, c.rope_base, &m.rope);
    let (cos, sin) = (c1.repeat(batch), s1.repeat(batch));
    let pass = Pass {
        m,
        rows,
        batch,
        t,
        threads,
        tables: (&cos[..], &sin[..]),
    };
    let mut x: Vec<f32> = ids
        .iter()
        .flat_map(|&i| m.tok[i * d..(i + 1) * d].iter().copied())
        .collect();
    let mut caches = Vec::with_capacity(m.blocks.len());
    for b in &m.blocks {
        let (out, cache) = pass.forward_block(b, &x);
        x = out;
        caches.push(cache);
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
        dx = pass.backward_block(b, ca, &dx, &mut g.blocks[bi]);
    }
    // Embedding gradient: rows summed per token id, in row order.
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..d {
            g.tok[id * d + i] += dx[r * d + i];
        }
    }
    Ok((loss, g))
}
