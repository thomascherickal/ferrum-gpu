use ferrum_ref::llama::{Config, RopeType};
use ferrum_wgpu::llama_train::param_sizes;

pub fn shape(
    (vocab, ctx, dim): (usize, usize, usize),
    (heads, kv_heads, head_dim): (usize, usize, usize),
    (layers, hidden): (usize, usize),
) -> Config {
    Config {
        vocab,
        ctx,
        dim,
        heads,
        kv_heads,
        head_dim,
        layers,
        hidden,
        rope_dim: head_dim,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

pub fn params(c: &Config, biases: bool) -> usize {
    param_sizes(c, biases).iter().sum()
}

/// The three sizes that are run for real.
pub fn sizes() -> [(&'static str, Config); 3] {
    [
        ("small", shape((1024, 128, 96), (4, 2, 24), (3, 256))),
        ("medium", shape((4096, 256, 256), (8, 2, 32), (6, 704))),
        ("large", shape((8192, 256, 512), (8, 4, 64), (8, 1408))),
    ]
}
