use super::*;

/// [`loss_and_grads`] with optional FFN-hidden dropout (see [`crate::dropout`]): the loss and the
/// gradient are those of the network with that step's mask applied.
pub fn loss_and_grads_with(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
    drop: Option<Dropout>,
) -> Result<(f32, Gpt2), String> {
    let t = check(m, ids, targets, batch)?;
    let (c, rows) = (&m.cfg, ids.len());
    let d = c.dim;
    let pass = Pass {
        m,
        rows,
        batch,
        t,
        threads,
        drop,
    };
    // Forward with caches.
    let mut x = nn::embed(&m.tok, &m.pos, ids, t, d);
    let mut caches = Vec::with_capacity(m.blocks.len());
    for (bi, b) in m.blocks.iter().enumerate() {
        let (out, cache) = pass.forward_block(bi, b, &x);
        x = out;
        caches.push(cache);
    }
    let hf = nn::layernorm(&x, &m.lnf_g, &m.lnf_b, rows, d);
    let logits = mm(&hf, &m.head, (rows, d, c.vocab), threads);
    let (loss, dlogits) = nn::cross_entropy(&logits, targets, c.vocab);

    // Backward.
    let mut g = m.zeros_like();
    let (dhf, dhead, _) = linear_bwd(&hf, &m.head, &dlogits, (rows, d, c.vocab), threads);
    g.head = dhead;
    let (mut dx, dg, db) = layernorm_bwd(&x, &m.lnf_g, &dhf, rows, d);
    g.lnf_g = dg;
    g.lnf_b = db;
    for (bi, (b, ca)) in m.blocks.iter().zip(&caches).enumerate().rev() {
        dx = pass.backward_block(b, ca, &dx, &mut g.blocks[bi]);
    }
    // Embedding gradients: rows are summed per token id and per position, in row order.
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..d {
            g.tok[id * d + i] += dx[r * d + i];
            g.pos[(r % t) * d + i] += dx[r * d + i];
        }
    }
    Ok((loss, g))
}
