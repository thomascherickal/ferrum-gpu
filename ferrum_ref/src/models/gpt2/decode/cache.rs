/// Per-layer key/value history for cached decoding (`[pos, dim]` each, grown as tokens arrive).
#[derive(Clone, Debug, Default)]
pub struct Cache {
    pub(super) k: Vec<Vec<f32>>,
    pub(super) v: Vec<Vec<f32>>,
}

impl Cache {
    /// An empty cache for a model with `layers` blocks.
    pub fn new(layers: usize) -> Self {
        Cache {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
        }
    }

    /// Tokens consumed so far.
    pub fn len(&self, dim: usize) -> usize {
        self.k.first().map_or(0, |k| k.len() / dim)
    }

    /// True when no token has been consumed.
    pub fn is_empty(&self) -> bool {
        self.k.first().is_none_or(|k| k.is_empty())
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.k
            .iter_mut()
            .chain(self.v.iter_mut())
            .for_each(Vec::clear);
    }
}

/// `y = x * w + bias` for one row, accumulating rows of `w` in ascending `k` (cache friendly).
pub(super) fn gemv(x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; n];
    for (p, &xv) in x.iter().enumerate() {
        for (yv, &wv) in y.iter_mut().zip(&w[p * n..(p + 1) * n]) {
            *yv += xv * wv;
        }
    }
    if let Some(b) = bias {
        y.iter_mut().zip(b).for_each(|(v, b)| *v += b);
    }
    y
}
