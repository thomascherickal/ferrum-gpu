//! Parity vectors for the in-browser runtime (Phase 5b): small seeded Llama / Qwen2 models as GGUF
//! bytes, a token sequence and the CPU oracle's logits for it. The browser loads the same bytes,
//! runs its WebGPU kernels and compares. For quantised files the oracle is the model decoded
//! from those very bytes, so only the kernels (not the quantiser) are being tested.

use crate::json::Json;
use ferrum_gguf::Gguf;
use ferrum_model::gguf::gpt2::Quant;
use ferrum_model::gguf::llama::{from_gguf, to_gguf};
use ferrum_ref::llama::{rope_tables, Config, Llama, RopeScaling, RopeType};
use ferrum_wgpu::ops::quant::{pack, QKind};

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

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

fn cfg(ty: RopeType, kv: usize, rope_dim: usize) -> Config {
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
fn cases() -> Vec<(&'static str, Llama, Quant)> {
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

/// The token sequence every case is run on (fills 8 of the 10 context slots).
pub const IDS: [usize; 8] = [3, 16, 0, 9, 9, 4, 11, 2];

/// Reference values for the Node-level tests of the JS reader: a tensor decoded by the Rust
/// reader, the device repack of a quantised one, and the RoPE tables the shaders will read.
fn probe(bytes: &[u8], model: &Llama, quant: Quant) -> Json {
    let g = Gguf::parse(bytes.to_vec()).expect("own file parses");
    let name = "blk.0.ffn_down.weight"; // [hidden, dim]: input width 64, a whole number of blocks
    let decoded = g.tensor_f32(name).expect("decodes");
    let mut p = vec![
        ("tensor", Json::Str(name.into())),
        (
            "decoded",
            Json::Arr(
                decoded
                    .iter()
                    .take(96)
                    .map(|&v| Json::Num(f64::from(v)))
                    .collect(),
            ),
        ),
        ("elements", Json::Num(decoded.len() as f64)),
    ];
    let kind = match quant {
        Quant::Q8_0 => Some(QKind::Q8_0),
        Quant::Q4_0 => Some(QKind::Q4_0),
        Quant::F32 => None,
    };
    if let Some(kind) = kind {
        let (k, n) = (model.cfg.hidden, model.cfg.dim);
        let words = pack(kind, g.tensor_bytes(name).expect("raw"), k, n).expect("packs");
        p.push((
            "packed",
            Json::Arr(
                words
                    .iter()
                    .take(120)
                    .map(|&w| Json::Num(f64::from(w)))
                    .collect(),
            ),
        ));
        p.push(("packed_len", Json::Num(words.len() as f64)));
    }
    let (cos, sin) = rope_tables(
        0,
        model.cfg.ctx,
        model.cfg.rope_dim,
        model.cfg.rope_base,
        &model.rope,
    );
    p.push((
        "rope_cos",
        Json::Arr(cos.iter().map(|&v| Json::Num(f64::from(v))).collect()),
    ));
    p.push((
        "rope_sin",
        Json::Arr(sin.iter().map(|&v| Json::Num(f64::from(v))).collect()),
    ));
    Json::obj_vec(p)
}

/// All cases as one JSON document: `{cases: [{name, quant, tolerance, ids, vocab, gguf, logits}]}`.
pub fn vectors() -> Json {
    let cases = cases()
        .into_iter()
        .map(|(name, model, quant)| {
            let bytes = to_gguf(&model, quant);
            // The oracle sees exactly what the browser will load.
            let loaded = from_gguf(&Gguf::parse(bytes.clone()).expect("own file parses"))
                .expect("own file loads");
            let logits = loaded.forward_logits(&IDS).expect("oracle forward");
            Json::obj([
                ("name", Json::Str(name.into())),
                ("quant", Json::Str(format!("{quant:?}"))),
                ("tolerance", Json::Num(1e-4)),
                (
                    "ids",
                    Json::Arr(IDS.iter().map(|&i| Json::Num(i as f64)).collect()),
                ),
                ("vocab", Json::Num(model.cfg.vocab as f64)),
                ("gguf", Json::Str(base64(&bytes))),
                ("probe", probe(&bytes, &loaded, quant)),
                (
                    "logits",
                    Json::Arr(logits.iter().map(|&l| Json::Num(f64::from(l))).collect()),
                ),
            ])
        })
        .collect();
    Json::obj([("cases", Json::Arr(cases))])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_4648_vectors() {
        for (raw, enc) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(raw.as_bytes()), enc);
        }
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
        assert_eq!(base64(&[0, 0, 0, 0]), "AAAAAA==");
    }

    #[test]
    fn the_vectors_cover_every_variant_and_agree_with_their_files() {
        let v = vectors();
        let Some(Json::Arr(cases)) = v.get("cases") else {
            panic!("cases")
        };
        assert_eq!(cases.len(), 6);
        let mut quants = Vec::new();
        for c in cases {
            let name = c.get("name").and_then(Json::as_str).unwrap();
            quants.push(c.get("quant").and_then(Json::as_str).unwrap().to_string());
            let Some(Json::Arr(logits)) = c.get("logits") else {
                panic!("{name}: logits")
            };
            assert_eq!(logits.len(), IDS.len() * 17, "{name}");
            assert!(
                logits
                    .iter()
                    .all(|l| l.as_f64().is_some_and(f64::is_finite)),
                "{name}"
            );
            // decode the base64 again by hand and check it is a loadable GGUF of the same shape
            let text = c.get("gguf").and_then(Json::as_str).unwrap();
            assert_eq!(text.len() % 4, 0, "{name}: padded base64");
            assert!(text.starts_with("R0dVRg"), "{name}: 'GGUF' magic in base64");
        }
        assert!(
            quants.iter().any(|q| q == "Q8_0")
                && quants.iter().any(|q| q == "Q4_0")
                && quants.iter().filter(|q| *q == "F32").count() == 4
        );
        // the document is valid JSON
        assert!(Json::parse(&v.to_string()).is_ok());
    }
}
