pub(super) fn clamp_prob(p: f32) -> f32 {
    if p.is_nan() {
        p
    } else {
        p.max(1e-12)
    }
}

/// Mean softmax cross-entropy and its gradient `(p - onehot) / rows`. Targets must be `< vocab`.
pub fn cross_entropy(logits: &[f32], targets: &[usize], vocab: usize) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let inv = 1.0 / rows as f32;
    let mut grad = vec![0.0; rows * vocab];
    let mut total = 0.0;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        total += -clamp_prob(probs[t]).ln();
        for j in 0..vocab {
            grad[i * vocab + j] = (probs[j] - if j == t { 1.0 } else { 0.0 }) * inv;
        }
    }
    (total * inv, grad)
}

/// Per row: `(log-probability of the target, whether the target is the first argmax)`.
pub fn score(logits: &[f32], targets: &[usize], vocab: usize) -> Vec<(f32, bool)> {
    targets
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
            let first_max = row.iter().position(|&x| x == max).unwrap_or(0);
            (row[t] - max - lse, first_max == t)
        })
        .collect()
}

/// `out[r] = tok[ids[r]] + pos[r % t]` for `ids.len()` rows of width `c`.
pub fn embed(tok: &[f32], pos: &[f32], ids: &[usize], t: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; ids.len() * c];
    for (r, &id) in ids.iter().enumerate() {
        for d in 0..c {
            out[r * c + d] = tok[id * c + d] + pos[(r % t) * c + d];
        }
    }
    out
}
