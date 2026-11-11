//! Token-level sampling from a logits row (the same semantics ferrum-cpu documents:
//! temperature 0 is exact argmax with the first maximum winning).

/// The shared seeded generator (lives in `ferrum_ref`, re-exported for callers of this module).
pub use ferrum_ref::rng::Rng;

/// Sampling controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// `0.0` = greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (`0` = off).
    pub top_k: usize,
    /// Keep the smallest set whose probability reaches `top_p` (`1.0` = off).
    pub top_p: f32,
    /// Divide positive (multiply negative) logits of recent tokens by this (`1.0` = off).
    pub repetition_penalty: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Index of the first maximum (NaN never wins).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] || logits[best].is_nan() && !v.is_nan() {
            best = i;
        }
    }
    best
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const L: [f32; 5] = [1.0, 3.0, 3.0, 0.5, -2.0];

    #[test]
    fn argmax_takes_the_first_maximum_and_ignores_nan() {
        assert_eq!(argmax(&L), 1);
        assert_eq!(argmax(&[f32::NAN, 2.0, 1.0]), 1);
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn temperature_zero_is_greedy_whatever_the_rng() {
        let mut r = Rng::new(1);
        for _ in 0..10 {
            assert_eq!(sample(&L, &Params::default(), &[], &mut r), 1);
        }
    }

    #[test]
    fn top_k_one_equals_greedy() {
        let p = Params {
            temperature: 1.5,
            top_k: 1,
            ..Params::default()
        };
        let mut r = Rng::new(2);
        for _ in 0..20 {
            assert_eq!(sample(&L, &p, &[], &mut r), 1);
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_covers_the_support() {
        let p = Params {
            temperature: 1.0,
            ..Params::default()
        };
        let run = |seed| {
            let mut r = Rng::new(seed);
            (0..200)
                .map(|_| sample(&L, &p, &[], &mut r))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(9), run(9));
        let draws = run(9);
        assert!(
            draws.contains(&1) && draws.contains(&2),
            "both tied maxima are drawn"
        );
        assert!(draws.iter().all(|&t| t < 5));
        let top = draws.iter().filter(|&&t| t == 1 || t == 2).count();
        assert!(
            top > 120,
            "most mass sits on the top two logits, got {top}/200"
        );
    }

    #[test]
    fn top_p_cuts_the_tail() {
        let p = Params {
            temperature: 1.0,
            top_p: 0.5,
            ..Params::default()
        };
        let mut r = Rng::new(3);
        for _ in 0..200 {
            let t = sample(&L, &p, &[], &mut r);
            assert!(t == 1 || t == 2, "token {t} is outside the nucleus");
        }
    }

    #[test]
    fn repetition_penalty_demotes_recent_tokens() {
        let p = Params {
            repetition_penalty: 4.0,
            ..Params::default()
        };
        let mut r = Rng::new(4);
        assert_eq!(sample(&L, &p, &[1], &mut r), 2); // 3.0 / 4 < 3.0, the tied twin wins
                                                     // Negative logits are pushed further down, not up.
        let neg = [-1.0, -1.5];
        assert_eq!(sample(&neg, &p, &[0], &mut r), 1);
        // Out-of-range recent tokens are ignored.
        assert_eq!(sample(&L, &p, &[99], &mut r), 1);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let mut r = Rng::new(5);
        let hot = Params {
            temperature: 1.0,
            ..Params::default()
        };
        assert_eq!(sample(&[f32::NAN, f32::NAN], &hot, &[], &mut r), 0);
        assert_eq!(sample(&[0.0], &hot, &[], &mut r), 0);
        let big = Params {
            temperature: 1.0,
            top_k: 100,
            top_p: 0.0001,
            ..Params::default()
        };
        assert_eq!(sample(&L, &big, &[], &mut r), 1);
    }
}
