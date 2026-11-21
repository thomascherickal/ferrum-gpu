use super::*;

/// Layer `i` of the file: its norms, attention (with optional q/k/v biases) and SwiGLU matrices.
pub(super) fn load_block(g: &Gguf, i: usize, cfg: &Config) -> Result<Block, GgufError> {
    let (d, h, qd, kvd) = (cfg.dim, cfg.hidden, cfg.q_dim(), cfg.kv_dim());
    let n = |s: &str| format!("blk.{i}.{s}");
    for stem in ["attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
        if g.tensor(&n(&format!("{stem}.bias"))).is_some() {
            return Err(err(format!(
                "tensor '{}' is not supported: biases are loaded only for attn_q/k/v",
                n(&format!("{stem}.bias"))
            )));
        }
    }
    Ok(Block {
        attn_norm: load(g, &n("attn_norm.weight"), &[d])?,
        wq: load_matrix(g, &n("attn_q.weight"), d, qd)?,
        bq: load_bias(g, &n("attn_q.bias"), qd)?,
        wk: load_matrix(g, &n("attn_k.weight"), d, kvd)?,
        bk: load_bias(g, &n("attn_k.bias"), kvd)?,
        wv: load_matrix(g, &n("attn_v.weight"), d, kvd)?,
        bv: load_bias(g, &n("attn_v.bias"), kvd)?,
        wo: load_matrix(g, &n("attn_output.weight"), qd, d)?,
        ffn_norm: load(g, &n("ffn_norm.weight"), &[d])?,
        gate: load_matrix(g, &n("ffn_gate.weight"), d, h)?,
        up: load_matrix(g, &n("ffn_up.weight"), d, h)?,
        down: load_matrix(g, &n("ffn_down.weight"), h, d)?,
    })
}
