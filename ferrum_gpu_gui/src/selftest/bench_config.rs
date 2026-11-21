use super::*;

/// The three decode-benchmark shapes of `ferrum_wgpu/benches/llama.rs` (random weights).
pub fn bench_config(shape: &str) -> Option<Config> {
    let (vocab, ctx, dim, heads, kv, hd, layers, hidden) = match shape {
        "small" => (1024, 128, 96, 4, 2, 24, 3, 256),
        "medium" => (4096, 256, 256, 8, 2, 32, 6, 704),
        "large" => (8192, 256, 512, 8, 4, 64, 8, 1408),
        _ => return None,
    };
    Some(Config {
        vocab,
        ctx,
        dim,
        heads,
        kv_heads: kv,
        head_dim: hd,
        layers,
        hidden,
        rope_dim: hd,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    })
}

/// A benchmark model as GGUF bytes (`quant` is `f32`, `q8_0` or `q4_0`).
pub fn bench_gguf(shape: &str, quant: &str) -> Result<Vec<u8>, String> {
    let cfg = bench_config(shape)
        .ok_or_else(|| format!("unknown shape '{shape}' (small, medium, large)"))?;
    let q = match quant {
        "f32" => Quant::F32,
        "q8_0" => Quant::Q8_0,
        "q4_0" => Quant::Q4_0,
        other => return Err(format!("unknown quant '{other}' (f32, q8_0, q4_0)")),
    };
    Ok(to_gguf(&Llama::random(cfg, 1, false, false), q))
}

/// The token sequence every case is run on (fills 8 of the 10 context slots).
pub const IDS: [usize; 8] = [3, 16, 0, 9, 9, 4, 11, 2];
