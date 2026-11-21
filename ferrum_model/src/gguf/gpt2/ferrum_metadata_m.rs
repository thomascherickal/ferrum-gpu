use super::*;

/// The generic `ferrum.metadata` JSON ferrum-cpu reads from every model file. Its shape follows a
/// ferrum-cpu `gpt2` file (fixtures/bpe/model.gguf); the fields a language model does not use are
/// the same constants.
pub(super) fn ferrum_metadata(c: &Config) -> String {
    let names: Vec<String> = (0..c.ctx).map(|i| format!("\"c_{i}\"")).collect();
    let range = format!("[0.000000,{:.6}]", c.vocab as f64);
    let ranges = vec![range.clone(); c.ctx].join(",");
    format!(
        "{{\"dataset_name\":\"ferrum-gpu Transformer\",\"task\":\"transformer_slm\",\"feature_names\":[{}],\"feature_ranges\":[{ranges}],\"class_names\":[],\"target_name\":\"next_char\",\"target_range\":{range},\"input_dim\":{},\"output_dim\":{},\"tokenizer_state\":\"\"}}",
        names.join(","),
        c.ctx,
        c.vocab
    )
}

/// Serialise as a `gpt2` GGUF. The tied head is omitted when it equals the transposed
/// embedding. With a tokenizer the file carries the `tokenizer.ggml.*` vocabulary (so llama.cpp
/// tokenizes the same way) and the `ferrum.*` keys ferrum-cpu needs to load it; the tokenizer's
/// vocabulary size must equal the model's.
pub fn to_gguf_with(m: &Gpt2, quant: Quant, tok: Option<&ferrum_tok::ByteBpeTokenizer>) -> Vec<u8> {
    let (c, d) = (&m.cfg, m.cfg.dim);
    let mut o = Out {
        b: GgufBuilder::new(),
        quant,
    };
    o.b.meta("general.architecture", Value::Str("gpt2".into()));
    o.b.meta("general.name", Value::Str("ferrum-gpu Transformer".into()));
    o.b.meta("general.quantization_version", Value::U32(2));
    o.b.meta("ferrum.format", Value::U32(1));
    o.b.meta("ferrum.quant", Value::Str(quant.tag().into()));
    o.b.meta("ferrum.normalizer", Value::Str(String::new()));
    o.b.meta("ferrum.metadata", Value::Str(ferrum_metadata(c)));
    if let Some(t) = tok {
        crate::tokenizer_gguf::put_vocab(&mut o.b, t);
    }
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
