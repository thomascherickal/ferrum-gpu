use super::*;

pub(super) const B64: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub(super) fn cfg(ty: RopeType, kv: usize, rope_dim: usize) -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: kv,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

/// The cases: `(name, model, on-disk precision)`.
pub(super) fn cases() -> Vec<(&'static str, Llama, Quant)> {
    let mut scaled = Llama::random(cfg(RopeType::Norm, 2, 8), 5, false, false);
    scaled.rope = RopeScaling {
        freq_factors: vec![1.0, 2.0, 4.0, 8.0],
        linear_factor: 2.0,
    };
    vec![
        (
            "llama f32",
            Llama::random(cfg(RopeType::Norm, 2, 8), 1, false, false),
            Quant::F32,
        ),
        (
            "qwen2 f32 (NEOX, q/k/v biases, tied head)",
            Llama::random(cfg(RopeType::Neox, 2, 8), 2, true, true),
            Quant::F32,
        ),
        (
            "llama f32 (MQA, partial rotary)",
            Llama::random(cfg(RopeType::Norm, 1, 4), 3, false, false),
            Quant::F32,
        ),
        ("llama f32 (scaled RoPE)", scaled, Quant::F32),
        (
            "llama Q8_0",
            Llama::random(cfg(RopeType::Norm, 2, 8), 4, false, false),
            Quant::Q8_0,
        ),
        (
            "qwen2 Q4_0 (biases)",
            Llama::random(cfg(RopeType::Neox, 2, 8), 6, true, false),
            Quant::Q4_0,
        ),
    ]
}

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
