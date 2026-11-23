use ferrum_ref::gpt2::Config;

/// Parameter count of a GPT-2 shape (tied-free: two vocabulary tables).
pub(crate) fn params(c: &Config) -> usize {
    let (d, h) = (c.dim, c.hidden);
    c.vocab * d * 2 + c.ctx * d + c.layers * (4 * d * d + 2 * d * h)
}

fn config(
    vocab: usize,
    ctx: usize,
    dim: usize,
    heads: usize,
    layers: usize,
    hidden: usize,
) -> Config {
    Config {
        vocab,
        ctx,
        dim,
        heads,
        layers,
        hidden,
    }
}

pub(crate) fn all() -> [(&'static str, Config); 3] {
    [
        ("small (reference preset)", config(1024, 128, 96, 4, 3, 320)),
        ("medium", config(4096, 256, 256, 8, 6, 1024)),
        ("large", config(8192, 256, 512, 8, 8, 2048)),
    ]
}
