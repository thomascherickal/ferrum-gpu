use super::*;

pub(super) fn check(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
) -> Result<usize, String> {
    let c = &m.cfg;
    if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
        return Err(format!(
            "{} tokens do not split into {batch} sequences",
            ids.len()
        ));
    }
    let t = ids.len() / batch;
    if t > c.ctx {
        return Err(format!("sequence length {t} exceeds the context {}", c.ctx));
    }
    if targets.len() != ids.len() {
        return Err(format!(
            "{} targets for {} tokens",
            targets.len(),
            ids.len()
        ));
    }
    if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= c.vocab) {
        return Err(format!("token {bad} is out of range for vocab {}", c.vocab));
    }
    Ok(t)
}

/// Mean next-token cross-entropy over all positions and the gradient of every parameter
/// (returned as a [`Gpt2`] of the same shape). `ids[i]` is predicted to be followed by
/// `targets[i]`. `threads` only splits matmuls, so the result is thread-count independent.
pub fn loss_and_grads(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
) -> Result<(f32, Gpt2), String> {
    loss_and_grads_with(m, ids, targets, batch, threads, None)
}

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
    let (d, hid, hd) = (c.dim, c.hidden, c.head_dim());
    // Forward with caches.
    let mut x = nn::embed(&m.tok, &m.pos, ids, t, d);
    let mut caches = Vec::with_capacity(m.blocks.len());
    for (bi, b) in m.blocks.iter().enumerate() {
        let x_in = x.clone();
        let h1 = nn::layernorm(&x, &b.ln1_g, &b.ln1_b, rows, d);
        let q = lin(&h1, &b.wq, &b.bq, (rows, d, d), threads);
        let k = lin(&h1, &b.wk, &b.bk, (rows, d, d), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, d), threads);
        let (a, probs) = nn::attention(&q, &k, &v, (batch, t, c.heads, hd));
        let x2 = nn::add(&x, &lin(&a, &b.wo, &b.bo, (rows, d, d), threads));
        let h2 = nn::layernorm(&x2, &b.ln2_g, &b.ln2_b, rows, d);
        let f1 = lin(&h2, &b.w1, &b.b1, (rows, d, hid), threads);
        let act: Vec<f32> = f1.iter().map(|&v| nn::gelu(v)).collect();
        let mask = drop.map(|d| d.mask(bi, act.len()));
        let act: Vec<f32> = match &mask {
            Some(mk) => act.iter().zip(mk).map(|(a, m)| a * m).collect(),
            None => act,
        };
        x = nn::add(&x2, &lin(&act, &b.w2, &b.b2, (rows, hid, d), threads));
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
            f1,
            act,
            mask,
        });
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
        let gb: &mut Block = &mut g.blocks[bi];
        // x_out = x2 + f2
        let (dact, dw2, db2) = linear_bwd(&ca.act, &b.w2, &dx, (rows, hid, d), threads);
        gb.w2 = dw2;
        gb.b2 = db2;
        // through the dropout mask (a dropped unit passes no gradient), then through GELU
        let df1: Vec<f32> = dact
            .iter()
            .zip(&ca.f1)
            .enumerate()
            .map(|(i, (&da, &f))| da * ca.mask.as_ref().map_or(1.0, |m| m[i]) * gelu_grad(f))
            .collect();
        let (dh2, dw1, db1) = linear_bwd(&ca.h2, &b.w1, &df1, (rows, d, hid), threads);
        gb.w1 = dw1;
        gb.b1 = db1;
        let (dx2_ln, dg2, dbl2) = layernorm_bwd(&ca.x2, &b.ln2_g, &dh2, rows, d);
        gb.ln2_g = dg2;
        gb.ln2_b = dbl2;
        let dx2 = nn::add(&dx, &dx2_ln);
        // x2 = x + o
        let (da, dwo, dbo) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, d, d), threads);
        gb.wo = dwo;
        gb.bo = dbo;
        let (dq, dk, dv) =
            attention_bwd(&ca.q, &ca.k, &ca.v, &ca.probs, &da, (batch, t, c.heads, hd));
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, d), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, d), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, d), threads);
        gb.wq = dwq;
        gb.bq = dbq;
        gb.wk = dwk;
        gb.bk = dbk;
        gb.wv = dwv;
        gb.bv = dbv;
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_ln, dg1, dbl1) = layernorm_bwd(&ca.x, &b.ln1_g, &dh1, rows, d);
        gb.ln1_g = dg1;
        gb.ln1_b = dbl1;
        dx = nn::add(&dx2, &dx_ln);
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
