use super::*;

/// Pick the next token. `recent` are the tokens the repetition penalty applies to.
pub fn sample(logits: &[f32], p: &Params, recent: &[usize], rng: &mut Rng) -> usize {
    let mut l = logits.to_vec();
    if p.repetition_penalty != 1.0 {
        for &t in recent {
            if let Some(v) = l.get_mut(t) {
                *v = if *v > 0.0 {
                    *v / p.repetition_penalty
                } else {
                    *v * p.repetition_penalty
                };
            }
        }
    }
    if p.temperature <= 0.0 {
        return argmax(&l);
    }
    // Candidates by descending logit; ties keep the lower index first (stable sort).
    let mut idx: Vec<usize> = (0..l.len()).filter(|&i| !l[i].is_nan()).collect();
    if idx.is_empty() {
        return 0;
    }
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    if p.top_k > 0 {
        idx.truncate(p.top_k);
    }
    let max = l[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((l[i] - max) / p.temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        let keep = probs
            .iter()
            .position(|&x| {
                acc += x;
                acc >= p.top_p
            })
            .map_or(probs.len(), |i| i + 1);
        idx.truncate(keep);
        probs.truncate(keep);
        let s: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= s);
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    // The last candidate is the fallback when rounding leaves the cumulative sum just under 1.
    let last = *idx.last().expect("non-empty");
    idx.iter()
        .zip(&probs)
        .find(|(_, &pr)| {
            acc += pr;
            r < acc
        })
        .map_or(last, |(&i, _)| i)
}
