use super::*;

/// The metadata of `m`, in file order.
pub(super) fn metadata(m: &Llama) -> Meta {
    let c = &m.cfg;
    let arch = match c.rope_type {
        RopeType::Norm => "llama",
        RopeType::Neox => "qwen2",
    };
    let u = |v: usize| Value::U32(v as u32);
    let mut kv = vec![
        ("general.architecture".to_string(), Value::Str(arch.into())),
        (format!("{arch}.context_length"), u(c.ctx)),
        (format!("{arch}.embedding_length"), u(c.dim)),
        (format!("{arch}.block_count"), u(c.layers)),
        (format!("{arch}.feed_forward_length"), u(c.hidden)),
        (format!("{arch}.attention.head_count"), u(c.heads)),
        (format!("{arch}.attention.head_count_kv"), u(c.kv_heads)),
        (format!("{arch}.attention.key_length"), u(c.head_dim)),
        (format!("{arch}.rope.dimension_count"), u(c.rope_dim)),
        (format!("{arch}.rope.freq_base"), Value::F32(c.rope_base)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Value::F32(c.eps),
        ),
    ];
    if m.rope.linear_factor != 1.0 {
        kv.push((
            format!("{arch}.rope.scaling.type"),
            Value::Str("linear".into()),
        ));
        kv.push((
            format!("{arch}.rope.scaling.factor"),
            Value::F32(m.rope.linear_factor),
        ));
    }
    kv
}

/// Serialise `m`, applying `edit` to the metadata and omitting the tensors named in `skip`
/// (the hostile-file tests use both; [`to_gguf`] uses neither).
pub(crate) fn write(m: &Llama, quant: Quant, edit: &dyn Fn(&mut Meta), skip: &[&str]) -> Vec<u8> {
    let (c, d) = (&m.cfg, m.cfg.dim);
    let mut o = Out {
        b: GgufBuilder::new(),
        quant,
        skip: skip.iter().map(|s| s.to_string()).collect(),
    };
    let mut kv = metadata(m);
    edit(&mut kv);
    kv.into_iter().for_each(|(k, v)| o.b.meta(&k, v));
    o.rows("token_embd.weight", &m.tok, c.vocab, d);
    for (i, b) in m.blocks.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}");
        o.vec(&n("attn_norm.weight"), &b.attn_norm);
        o.matrix(&n("attn_q.weight"), &b.wq, d, c.q_dim());
        o.bias(&n("attn_q.bias"), &b.bq);
        o.matrix(&n("attn_k.weight"), &b.wk, d, c.kv_dim());
        o.bias(&n("attn_k.bias"), &b.bk);
        o.matrix(&n("attn_v.weight"), &b.wv, d, c.kv_dim());
        o.bias(&n("attn_v.bias"), &b.bv);
        o.matrix(&n("attn_output.weight"), &b.wo, c.q_dim(), d);
        o.vec(&n("ffn_norm.weight"), &b.ffn_norm);
        o.matrix(&n("ffn_gate.weight"), &b.gate, d, c.hidden);
        o.matrix(&n("ffn_up.weight"), &b.up, d, c.hidden);
        o.matrix(&n("ffn_down.weight"), &b.down, c.hidden, d);
    }
    o.vec("output_norm.weight", &m.out_norm);
    if !m.tied {
        o.matrix("output.weight", &m.head, d, c.vocab);
    }
    if !m.rope.freq_factors.is_empty() {
        o.vec("rope_freqs.weight", &m.rope.freq_factors);
    }
    o.b.into_bytes()
}

/// Serialise as a `llama` (NORM RoPE) or `qwen2` (NEOX RoPE) GGUF. A tied model omits
/// `output.weight`. No tokenizer metadata is written (Phase 6).
pub fn to_gguf(m: &Llama, quant: Quant) -> Vec<u8> {
    write(m, quant, &|_| {}, &[])
}

/// Like [`to_gguf`], plus extra metadata (a `tokenizer.ggml.*` table, a chat template).
pub fn to_gguf_with_meta(m: &Llama, quant: Quant, extra: &[(String, Value)]) -> Vec<u8> {
    write(m, quant, &|kv| kv.extend(extra.iter().cloned()), &[])
}
