/// `shader_sources!("group/name", ..)`: `(name, source)` for each kernel file; `include_str!`
/// needs a literal path, so the macro builds it from the name.
macro_rules! shader_sources {
    ($($name:literal),* $(,)?) => {
        &[$(($name, include_str!(concat!($name, ".wgsl")))),*]
    };
}

/// `(group/name, source)` for every kernel under `src/shaders/`.
#[rustfmt::skip]
pub static SOURCES: &[(&str, &str)] = shader_sources![
    "bwd/attn_bwd_dk", "bwd/attn_bwd_dot", "bwd/attn_bwd_dp", "bwd/attn_bwd_dq",
    "bwd/attn_bwd_ds", "bwd/attn_bwd_dv", "bwd/colsum", "bwd/embed_bwd_pos",
    "bwd/embed_bwd_tok", "bwd/gelu_bwd", "bwd/ln_bwd_dx", "bwd/ln_bwd_params",
    "fwd/add", "fwd/attn_dec_probs", "fwd/attn_dec_pv", "fwd/attn_pv", "fwd/attn_scores",
    "fwd/attn_softmax", "fwd/bias_add", "fwd/ce_rows", "fwd/dropout", "fwd/embed", "fwd/gelu",
    "fwd/gemv", "fwd/layernorm", "fwd/mean_loss", "fwd/score",
    "llama/attn_gqa_bwd_dk", "llama/attn_gqa_bwd_dp", "llama/attn_gqa_bwd_dq",
    "llama/attn_gqa_bwd_dv", "llama/attn_gqa_pv", "llama/attn_gqa_scores",
    "llama/attn_gqd_probs", "llama/attn_gqd_pv", "llama/gather", "llama/rms_bwd_dx",
    "llama/rms_bwd_gain", "llama/rmsnorm", "llama/rope", "llama/swiglu", "llama/swiglu_bwd",
    "matmul/matmul_blocked4", "matmul/matmul_blocked4t", "matmul/matmul_naive",
    "matmul/matmul_tiled16",
    "opt/accumulate", "opt/adam", "opt/norm_final", "opt/sumsq_part", "opt/transpose",
    "quant/q8_snap", "quant/qgemv", "quant/qlinear", "quant/qlinear_ext", "quant/qlinear_tiled",
];
