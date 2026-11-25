use super::*;

pub(super) struct BlockCache {
    pub(super) x: Vec<f32>,
    pub(super) h1: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) probs: Vec<f32>,
    pub(super) a: Vec<f32>,
    pub(super) x2: Vec<f32>,
    pub(super) h2: Vec<f32>,
    pub(super) gate: Vec<f32>,
    pub(super) up: Vec<f32>,
    pub(super) act: Vec<f32>,
}

pub(super) fn check(
    m: &Llama,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
) -> Result<usize, String> {
    let c = &m.cfg;
    if m.tied {
        return Err("training a tied head is not supported: train an untied copy".into());
    }
    crate::grad::check_batch(ids, targets, batch, c.ctx, c.vocab)
}

pub(super) fn rope_dims(c: &Config, rows: usize, heads: usize) -> (usize, usize, usize, usize) {
    (rows, heads, c.head_dim, c.rope_dim)
}
