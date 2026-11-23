use ferrum_ref::gpt2::Config;

/// `(name, model, batch, tokens per sequence, timed reps)`.
pub(crate) type Case = (&'static str, Config, usize, usize, usize);

pub(crate) fn all() -> [Case; 3] {
    let cfg = |vocab, dim, heads, layers, hidden| Config {
        vocab,
        ctx: 128,
        dim,
        heads,
        layers,
        hidden,
    };
    [
        (
            "small (reference preset)",
            cfg(1024, 96, 4, 3, 320),
            16,
            128,
            10,
        ),
        ("medium", cfg(4096, 256, 8, 4, 1024), 8, 128, 5),
        ("large", cfg(4096, 512, 8, 6, 2048), 8, 128, 3),
    ]
}
