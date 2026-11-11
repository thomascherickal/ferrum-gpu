// Provenance: tensor names, the QKV stacking and the transposition convention follow
// ferrum-cpu `ferrum_core/src/loader/gpt2.rs` and `loader/encode.rs` (commit 56a9d3a), which
// write llama.cpp's `gpt2` layout. The code is our own.
//! `gpt2` GGUF <-> [`Gpt2`].
//!
//! GGUF stores each matrix with GGML dim 0 (contiguous) = the input dimension, so a stored
//! matrix is `[out, in]` row-major; our weights are `[in, out]`, hence a transpose each way.
//! `attn_qkv.weight` stacks the Q, K, V matrices (each transposed) along the output axis.
//! A missing `output.weight` means the head is tied to `token_embd.weight`.

use ferrum_gguf::quant::encode;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value, GGML_F32, GGML_Q4_0, GGML_Q8_0, QK};
use ferrum_ref::gpt2::{Block, Config, Gpt2};
use ferrum_ref::nn::LN_EPS;

/// On-disk precision for matrices (vectors are always F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// 32-bit float.
    F32,
    /// Q8_0 where the input dimension is a multiple of 32, else F32.
    Q8_0,
    /// Q4_0 where the input dimension is a multiple of 32, else F32.
    Q4_0,
}

fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), GgufError> {
    if got == want {
        Ok(())
    } else {
        Err(err(format!(
            "tensor '{name}' has {got} elements, expected {want}"
        )))
    }
}

/// Read a tensor, check its element count, return its f32 data.
fn load(g: &Gguf, name: &str, want: usize) -> Result<Vec<f32>, GgufError> {
    let v = g.tensor_f32(name)?;
    expect_len(name, v.len(), want)?;
    Ok(v)
}

/// Load a stored `[out, in]` matrix as our `[in, out]`.
fn load_matrix(g: &Gguf, name: &str, inp: usize, out: usize) -> Result<Vec<f32>, GgufError> {
    Ok(transpose(&load(g, name, inp * out)?, out, inp))
}

/// Build a [`Gpt2`] from a `gpt2`-architecture GGUF.
pub fn from_gguf(g: &Gguf) -> Result<Gpt2, GgufError> {
    let arch = g.string("general.architecture")?;
    if arch != "gpt2" {
        return Err(err(format!(
            "architecture is '{arch}', only 'gpt2' is supported"
        )));
    }
    let eps = g.float("gpt2.attention.layer_norm_epsilon")?;
    if (eps - LN_EPS).abs() > 1e-9 {
        return Err(err(format!(
            "layer_norm_epsilon is {eps}; the kernels use {LN_EPS}"
        )));
    }
    let k = |s: &str| g.uint(&format!("gpt2.{s}"));
    let (d, ctx) = (k("embedding_length")?, k("context_length")?);
    let tok_info = g
        .tensor("token_embd.weight")
        .ok_or_else(|| err("missing tensor 'token_embd.weight'"))?;
    if tok_info.dims.len() != 2 || tok_info.dims[0] != d as u64 {
        return Err(err("token_embd.weight must be [embedding_length, vocab]"));
    }
    let cfg = Config {
        vocab: tok_info.dims[1] as usize,
        ctx,
        dim: d,
        heads: k("attention.head_count")?,
        layers: k("block_count")?,
        hidden: k("feed_forward_length")?,
    };
    cfg.validate().map_err(err)?;
    // Header numbers size every allocation below, so bound them by the file's own tensors:
    // the embedding tensor was already proven to fit inside the file by the parser.
    let tok = load(g, "token_embd.weight", cfg.vocab * d)?;
    let pos = load(g, "position_embd.weight", ctx * d)?;
    let mut blocks = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        let qkv = load(g, &n("attn_qkv.weight"), 3 * d * d)?;
        let qkv_b = load(g, &n("attn_qkv.bias"), 3 * d)?;
        let part = |j: usize| transpose(&qkv[j * d * d..(j + 1) * d * d], d, d);
        blocks.push(Block {
            ln1_g: load(g, &n("attn_norm.weight"), d)?,
            ln1_b: load(g, &n("attn_norm.bias"), d)?,
            wq: part(0),
            bq: qkv_b[..d].to_vec(),
            wk: part(1),
            bk: qkv_b[d..2 * d].to_vec(),
            wv: part(2),
            bv: qkv_b[2 * d..].to_vec(),
            wo: load_matrix(g, &n("attn_output.weight"), d, d)?,
            bo: load(g, &n("attn_output.bias"), d)?,
            ln2_g: load(g, &n("ffn_norm.weight"), d)?,
            ln2_b: load(g, &n("ffn_norm.bias"), d)?,
            w1: load_matrix(g, &n("ffn_up.weight"), d, cfg.hidden)?,
            b1: load(g, &n("ffn_up.bias"), cfg.hidden)?,
            w2: load_matrix(g, &n("ffn_down.weight"), cfg.hidden, d)?,
            b2: load(g, &n("ffn_down.bias"), d)?,
        });
    }
    let head = if g.tensor("output.weight").is_some() {
        load_matrix(g, "output.weight", d, cfg.vocab)?
    } else {
        transpose(&tok, cfg.vocab, d) // tied: head[in][out] = tok[out][in]
    };
    Ok(Gpt2 {
        cfg,
        lnf_g: load(g, "output_norm.weight", d)?,
        lnf_b: load(g, "output_norm.bias", d)?,
        tok,
        pos,
        blocks,
        head,
    })
}

struct Out {
    b: GgufBuilder,
    quant: Quant,
}

impl Out {
    fn vec(&mut self, name: &str, data: &[f32]) {
        self.b.tensor_f32(name, &[data.len() as u64], data);
    }

    /// `rows` GGML rows of `ne0` values, quantised per the precision when `ne0 % 32 == 0`.
    fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        let ty = match self.quant {
            Quant::Q8_0 if ne0.is_multiple_of(QK) => GGML_Q8_0,
            Quant::Q4_0 if ne0.is_multiple_of(QK) => GGML_Q4_0,
            _ => GGML_F32,
        };
        let bytes = encode(ty, data).expect("block-aligned by the check above");
        self.b
            .tensor_raw(name, &[ne0 as u64, rows as u64], ty, bytes);
    }

    /// Our `[in, out]` matrix, stored transposed.
    fn matrix(&mut self, name: &str, w: &[f32], inp: usize, out: usize) {
        self.rows(name, &transpose(w, inp, out), out, inp);
    }
}

/// Serialise as a `gpt2` GGUF. The tied head is omitted when it equals the transposed
/// embedding. No tokenizer metadata is written (Phase 6), so llama.cpp cannot run the file yet.
pub fn to_gguf(m: &Gpt2, quant: Quant) -> Vec<u8> {
    let (c, d) = (&m.cfg, m.cfg.dim);
    let mut o = Out {
        b: GgufBuilder::new(),
        quant,
    };
    o.b.meta("general.architecture", Value::Str("gpt2".into()));
    for (key, v) in [
        ("context_length", c.ctx),
        ("embedding_length", d),
        ("block_count", c.layers),
        ("feed_forward_length", c.hidden),
        ("attention.head_count", c.heads),
    ] {
        o.b.meta(&format!("gpt2.{key}"), Value::U32(v as u32));
    }
    o.b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    o.rows("token_embd.weight", &m.tok, c.vocab, d);
    o.b.tensor_f32("position_embd.weight", &[d as u64, c.ctx as u64], &m.pos);
    for (i, b) in m.blocks.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}");
        o.vec(&n("attn_norm.weight"), &b.ln1_g);
        o.vec(&n("attn_norm.bias"), &b.ln1_b);
        let qkv: Vec<f32> = [&b.wq, &b.wk, &b.wv]
            .iter()
            .flat_map(|w| transpose(w, d, d))
            .collect();
        o.rows(&n("attn_qkv.weight"), &qkv, 3 * d, d);
        o.vec(&n("attn_qkv.bias"), &[&b.bq[..], &b.bk, &b.bv].concat());
        o.matrix(&n("attn_output.weight"), &b.wo, d, d);
        o.vec(&n("attn_output.bias"), &b.bo);
        o.vec(&n("ffn_norm.weight"), &b.ln2_g);
        o.vec(&n("ffn_norm.bias"), &b.ln2_b);
        o.matrix(&n("ffn_up.weight"), &b.w1, d, c.hidden);
        o.vec(&n("ffn_up.bias"), &b.b1);
        o.matrix(&n("ffn_down.weight"), &b.w2, c.hidden, d);
        o.vec(&n("ffn_down.bias"), &b.b2);
    }
    o.vec("output_norm.weight", &m.lnf_g);
    o.vec("output_norm.bias", &m.lnf_b);
    let head_rows = transpose(&m.head, d, c.vocab);
    if head_rows != m.tok {
        o.rows("output.weight", &head_rows, c.vocab, d);
    }
    o.b.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dim: usize) -> Config {
        Config {
            vocab: 13,
            ctx: 8,
            dim,
            heads: 2,
            layers: 2,
            hidden: dim * 2,
        }
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn transpose_is_correct_and_an_involution() {
        assert_eq!(
            transpose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let m: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(transpose(&transpose(&m, 3, 4), 4, 3), m);
    }

    #[test]
    fn f32_round_trip_is_exact() {
        let m = Gpt2::random(cfg(8), 1);
        let back = from_gguf(&Gguf::parse(to_gguf(&m, Quant::F32)).unwrap()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn round_trip_gives_identical_logits() {
        let m = Gpt2::random(cfg(8), 2);
        let back = from_gguf(&Gguf::parse(to_gguf(&m, Quant::F32)).unwrap()).unwrap();
        let ids = [3, 1, 4, 1, 5];
        assert_eq!(
            m.forward_logits(&ids).unwrap(),
            back.forward_logits(&ids).unwrap()
        );
    }

    #[test]
    fn tied_head_is_omitted_and_restored() {
        let mut m = Gpt2::random(cfg(8), 3);
        m.head = transpose(&m.tok, m.cfg.vocab, m.cfg.dim);
        let bytes = to_gguf(&m, Quant::F32);
        let g = Gguf::parse(bytes).unwrap();
        assert!(
            g.tensor("output.weight").is_none(),
            "a tied head is not stored"
        );
        assert_eq!(from_gguf(&g).unwrap(), m);
        // An untied head is stored.
        let untied = Gpt2::random(cfg(8), 3);
        assert!(Gguf::parse(to_gguf(&untied, Quant::F32))
            .unwrap()
            .tensor("output.weight")
            .is_some());
    }

    #[test]
    fn stored_layout_matches_ggml_conventions() {
        let m = Gpt2::random(cfg(8), 4);
        let g = Gguf::parse(to_gguf(&m, Quant::F32)).unwrap();
        assert_eq!(g.tensor("token_embd.weight").unwrap().dims, vec![8, 13]);
        assert_eq!(g.tensor("position_embd.weight").unwrap().dims, vec![8, 8]);
        assert_eq!(g.tensor("blk.0.attn_qkv.weight").unwrap().dims, vec![8, 24]);
        assert_eq!(g.tensor("blk.1.ffn_up.weight").unwrap().dims, vec![8, 16]);
        assert_eq!(g.tensor("blk.1.ffn_down.weight").unwrap().dims, vec![16, 8]);
        assert_eq!(g.tensor("output.weight").unwrap().dims, vec![8, 13]);
        // GGML row 0 of attn_qkv is the first column of Wq (output unit 0 over all inputs).
        let qkv = g.tensor_f32("blk.0.attn_qkv.weight").unwrap();
        let wq = &m.blocks[0].wq;
        assert_eq!(qkv[3], wq[3 * 8]); // [out 0][in 3] = Wq[in 3][out 0]
        assert_eq!(qkv[8 + 2], wq[2 * 8 + 1]); // [out 1][in 2] = Wq[in 2][out 1]
                                               // The K block starts d rows later.
        assert_eq!(qkv[8 * 8 + 5], m.blocks[0].wk[5 * 8]);
    }

    #[test]
    fn quantised_models_load_and_stay_close() {
        let m = Gpt2::random(cfg(32), 5);
        let ids = [1, 7, 3, 9];
        let want = m.forward_logits(&ids).unwrap();
        let q8 = from_gguf(&Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap()).unwrap();
        let q4 = from_gguf(&Gguf::parse(to_gguf(&m, Quant::Q4_0)).unwrap()).unwrap();
        let (e8, e4) = (
            max_diff(&q8.forward_logits(&ids).unwrap(), &want),
            max_diff(&q4.forward_logits(&ids).unwrap(), &want),
        );
        assert!(e8 < 0.3, "q8 logits drift {e8}");
        assert!(e8 < e4, "q4 ({e4}) must be coarser than q8 ({e8})");
        // Ineligible widths (dim 8) stay F32 even when Q8_0 is requested.
        let small = Gpt2::random(cfg(8), 5);
        let g = Gguf::parse(to_gguf(&small, Quant::Q8_0)).unwrap();
        assert_eq!(g.tensor("blk.0.ffn_up.weight").unwrap().ggml_type, GGML_F32);
        // A 32-wide model stores its matrices quantised.
        let g32 = Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap();
        assert_eq!(
            g32.tensor("blk.0.ffn_up.weight").unwrap().ggml_type,
            GGML_Q8_0
        );
        assert_eq!(
            g32.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
            GGML_F32
        );
    }

    #[test]
    fn a_qat_snapped_model_survives_a_q8_0_export_exactly() {
        // The point of quantisation-aware training: after the snap, exporting as Q8_0 and
        // reloading loses nothing, because the snap and the export use the same quantiser.
        let snapped = Gpt2::random(cfg(32), 11).fake_quantize();
        let back = from_gguf(&Gguf::parse(to_gguf(&snapped, Quant::Q8_0)).unwrap()).unwrap();
        assert_eq!(back, snapped, "every tensor, bit for bit");
        // An unsnapped model of the same shape does lose precision on export.
        let raw = Gpt2::random(cfg(32), 11);
        let lossy = from_gguf(&Gguf::parse(to_gguf(&raw, Quant::Q8_0)).unwrap()).unwrap();
        assert_ne!(lossy, raw);
        assert_eq!(
            lossy,
            raw.fake_quantize(),
            "the export applies exactly the QAT snap"
        );
        // And the snapped model computes the same logits before and after the round trip.
        let ids = [3, 1, 4, 1, 5];
        assert_eq!(
            snapped.forward_logits(&ids).unwrap(),
            back.forward_logits(&ids).unwrap()
        );
    }

    #[test]
    fn rejects_foreign_or_inconsistent_files() {
        let m = Gpt2::random(cfg(8), 6);
        let ok = to_gguf(&m, Quant::F32);
        // wrong architecture
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("llama".into()));
        assert!(from_gguf(&Gguf::parse(b.into_bytes()).unwrap())
            .unwrap_err()
            .to_string()
            .contains("only 'gpt2'"));
        // missing metadata and tensors
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        assert!(from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).is_err());
        // a different epsilon
        let mut bytes = ok.clone();
        let eps = LN_EPS.to_bits().to_le_bytes();
        let at = bytes
            .windows(4)
            .position(|w| w == eps)
            .expect("epsilon bytes present");
        bytes[at..at + 4].copy_from_slice(&1e-6f32.to_bits().to_le_bytes());
        assert!(from_gguf(&Gguf::parse(bytes).unwrap())
            .unwrap_err()
            .to_string()
            .contains("layer_norm_epsilon"));
    }

    #[test]
    fn an_embedding_whose_width_disagrees_with_the_metadata_is_refused() {
        let m = Gpt2::random(cfg(8), 8);
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        for (k, v) in [
            ("context_length", 8u32),
            ("embedding_length", 8),
            ("block_count", 1),
            ("feed_forward_length", 16),
            ("attention.head_count", 2),
        ] {
            b.meta(&format!("gpt2.{k}"), Value::U32(v));
        }
        b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
        // Declared width 8, but the tensor's contiguous dimension is 4.
        b.tensor_f32("token_embd.weight", &[4, 26], &m.tok);
        let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
        assert!(
            e.to_string().contains("must be [embedding_length, vocab]"),
            "{e}"
        );
    }

    #[test]
    fn a_tensor_of_the_wrong_size_is_named() {
        let m = Gpt2::random(cfg(8), 7);
        // Same file but claim heads that do not divide the width.
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        for (k, v) in [
            ("context_length", 8u32),
            ("embedding_length", 8),
            ("block_count", 1),
            ("feed_forward_length", 16),
            ("attention.head_count", 3),
        ] {
            b.meta(&format!("gpt2.{k}"), Value::U32(v));
        }
        b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
        b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
        let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
        assert!(e.to_string().contains("divisible"), "{e}");
        // Right metadata, truncated position table.
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        for (k, v) in [
            ("context_length", 8u32),
            ("embedding_length", 8),
            ("block_count", 1),
            ("feed_forward_length", 16),
            ("attention.head_count", 2),
        ] {
            b.meta(&format!("gpt2.{k}"), Value::U32(v));
        }
        b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
        b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
        b.tensor_f32("position_embd.weight", &[8, 2], &m.pos[..16]);
        let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
        assert!(
            e.to_string().contains("position_embd.weight") && e.to_string().contains("expected 64"),
            "{e}"
        );
    }
}
