use ferrum_ref::llama::{Config, RopeType};

/// A real-size model shape to try on the adapter.
pub(crate) struct Shape {
    pub name: &'static str,
    pub cfg: Config,
    pub biases: bool,
    pub tied: bool,
}

fn shape(
    name: &'static str,
    (vocab, ctx, dim): (usize, usize, usize),
    (heads, kv_heads, head_dim): (usize, usize, usize),
    (layers, hidden): (usize, usize),
    (biases, tied): (bool, bool),
) -> Shape {
    let cfg = Config {
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
    };
    Shape {
        name,
        cfg,
        biases,
        tied,
    }
}

pub(crate) fn all() -> [Shape; 5] {
    [
        shape(
            "SmolLM-135M shape",
            (49152, 2048, 576),
            (9, 3, 64),
            (30, 1536),
            (false, true),
        ),
        shape(
            "SmolLM-360M shape",
            (49152, 2048, 960),
            (15, 5, 64),
            (32, 2560),
            (false, true),
        ),
        shape(
            "Qwen2-0.5B shape",
            (151936, 2048, 896),
            (14, 2, 64),
            (24, 4864),
            (true, true),
        ),
        shape(
            "TinyLlama-1.1B shape",
            (32000, 2048, 2048),
            (32, 4, 64),
            (22, 5632),
            (false, false),
        ),
        shape(
            "Qwen2.5-1.5B shape",
            (151936, 2048, 1536),
            (12, 2, 128),
            (28, 8960),
            (true, true),
        ),
    ]
}
