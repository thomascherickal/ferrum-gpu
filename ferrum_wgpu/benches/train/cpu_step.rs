use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::grad::loss_and_grads;
use ferrum_ref::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};

pub fn params(c: &Config) -> usize {
    Gpt2::random(Config { layers: 1, ..*c }, 1).num_params()
        + (c.layers - 1) * (4 * c.dim * c.dim + 2 * c.dim * c.hidden + 10 * c.dim + c.hidden)
}

/// One batch: token ids and their targets.
pub struct Batch<'a> {
    pub ids: &'a [usize],
    pub targets: &'a [usize],
    pub size: usize,
}

/// Forward and backward over `shards` shards of the batch, one thread each.
fn shard_grads(w: &Gpt2, b: &Batch, shards: usize) -> Vec<(f32, Gpt2)> {
    let per = b.size / shards;
    let t = b.ids.len() / b.size;
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..shards)
            .map(|i| {
                let (lo, hi) = (i * per * t, (i + 1) * per * t);
                s.spawn(move || {
                    loss_and_grads(w, &b.ids[lo..hi], &b.targets[lo..hi], per, 1)
                        .expect("valid batch")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("shard thread"))
            .collect()
    })
}

/// Data-parallel CPU step over `shards` shards; returns the mean loss.
pub fn cpu_step(w: &mut Gpt2, st: &mut AdamState, b: &Batch, shards: usize, adam: &Adam) -> f32 {
    let results = shard_grads(w, b, shards);
    let n = shards as f32;
    let loss = results.iter().map(|r| r.0).sum::<f32>() / n;
    let mut g = results[0].1.clone();
    for (i, p) in g.params_mut().into_iter().enumerate() {
        for r in &results[1..] {
            for (a, b) in p.iter_mut().zip(r.1.params()[i]) {
                *a += b;
            }
        }
        p.iter_mut().for_each(|a| *a /= n);
    }
    let scale = clip_scale(global_norm(&g), 1.0);
    adam_step(w, &g, st, adam, scale);
    loss
}
