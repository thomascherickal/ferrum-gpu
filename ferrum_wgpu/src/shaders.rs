//! Every WGSL kernel as source text, by `group/name`, plus the prelude the native runtime
//! prepends to each. This is what the browser runtime (Phase 5b) is served, so the page runs
//! the same source files as the native engine rather than a copy. A test keeps the list equal to
//! the files on disk and compiles each entry.

/// WGSL prepended to every kernel (the global index of a 1-D dispatch).
pub const PRELUDE: &str = crate::runtime::PRELUDE;

/// `(group/name, source)` for every kernel under `src/shaders/`.
pub static SOURCES: &[(&str, &str)] = &[
    (
        "bwd/attn_bwd_dk",
        include_str!("shaders/bwd/attn_bwd_dk.wgsl"),
    ),
    (
        "bwd/attn_bwd_dot",
        include_str!("shaders/bwd/attn_bwd_dot.wgsl"),
    ),
    (
        "bwd/attn_bwd_dp",
        include_str!("shaders/bwd/attn_bwd_dp.wgsl"),
    ),
    (
        "bwd/attn_bwd_dq",
        include_str!("shaders/bwd/attn_bwd_dq.wgsl"),
    ),
    (
        "bwd/attn_bwd_ds",
        include_str!("shaders/bwd/attn_bwd_ds.wgsl"),
    ),
    (
        "bwd/attn_bwd_dv",
        include_str!("shaders/bwd/attn_bwd_dv.wgsl"),
    ),
    ("bwd/colsum", include_str!("shaders/bwd/colsum.wgsl")),
    (
        "bwd/embed_bwd_pos",
        include_str!("shaders/bwd/embed_bwd_pos.wgsl"),
    ),
    (
        "bwd/embed_bwd_tok",
        include_str!("shaders/bwd/embed_bwd_tok.wgsl"),
    ),
    ("bwd/gelu_bwd", include_str!("shaders/bwd/gelu_bwd.wgsl")),
    ("bwd/ln_bwd_dx", include_str!("shaders/bwd/ln_bwd_dx.wgsl")),
    (
        "bwd/ln_bwd_params",
        include_str!("shaders/bwd/ln_bwd_params.wgsl"),
    ),
    ("fwd/add", include_str!("shaders/fwd/add.wgsl")),
    (
        "fwd/attn_dec_probs",
        include_str!("shaders/fwd/attn_dec_probs.wgsl"),
    ),
    (
        "fwd/attn_dec_pv",
        include_str!("shaders/fwd/attn_dec_pv.wgsl"),
    ),
    ("fwd/attn_pv", include_str!("shaders/fwd/attn_pv.wgsl")),
    (
        "fwd/attn_scores",
        include_str!("shaders/fwd/attn_scores.wgsl"),
    ),
    (
        "fwd/attn_softmax",
        include_str!("shaders/fwd/attn_softmax.wgsl"),
    ),
    ("fwd/bias_add", include_str!("shaders/fwd/bias_add.wgsl")),
    ("fwd/ce_rows", include_str!("shaders/fwd/ce_rows.wgsl")),
    ("fwd/embed", include_str!("shaders/fwd/embed.wgsl")),
    ("fwd/gelu", include_str!("shaders/fwd/gelu.wgsl")),
    ("fwd/gemv", include_str!("shaders/fwd/gemv.wgsl")),
    ("fwd/layernorm", include_str!("shaders/fwd/layernorm.wgsl")),
    ("fwd/mean_loss", include_str!("shaders/fwd/mean_loss.wgsl")),
    ("fwd/score", include_str!("shaders/fwd/score.wgsl")),
    (
        "llama/attn_gqa_bwd_dk",
        include_str!("shaders/llama/attn_gqa_bwd_dk.wgsl"),
    ),
    (
        "llama/attn_gqa_bwd_dp",
        include_str!("shaders/llama/attn_gqa_bwd_dp.wgsl"),
    ),
    (
        "llama/attn_gqa_bwd_dq",
        include_str!("shaders/llama/attn_gqa_bwd_dq.wgsl"),
    ),
    (
        "llama/attn_gqa_bwd_dv",
        include_str!("shaders/llama/attn_gqa_bwd_dv.wgsl"),
    ),
    (
        "llama/attn_gqa_pv",
        include_str!("shaders/llama/attn_gqa_pv.wgsl"),
    ),
    (
        "llama/attn_gqa_scores",
        include_str!("shaders/llama/attn_gqa_scores.wgsl"),
    ),
    (
        "llama/attn_gqd_probs",
        include_str!("shaders/llama/attn_gqd_probs.wgsl"),
    ),
    (
        "llama/attn_gqd_pv",
        include_str!("shaders/llama/attn_gqd_pv.wgsl"),
    ),
    ("llama/gather", include_str!("shaders/llama/gather.wgsl")),
    (
        "llama/rms_bwd_dx",
        include_str!("shaders/llama/rms_bwd_dx.wgsl"),
    ),
    (
        "llama/rms_bwd_gain",
        include_str!("shaders/llama/rms_bwd_gain.wgsl"),
    ),
    ("llama/rmsnorm", include_str!("shaders/llama/rmsnorm.wgsl")),
    ("llama/rope", include_str!("shaders/llama/rope.wgsl")),
    ("llama/swiglu", include_str!("shaders/llama/swiglu.wgsl")),
    (
        "llama/swiglu_bwd",
        include_str!("shaders/llama/swiglu_bwd.wgsl"),
    ),
    (
        "matmul/matmul_blocked4",
        include_str!("shaders/matmul/matmul_blocked4.wgsl"),
    ),
    (
        "matmul/matmul_blocked4t",
        include_str!("shaders/matmul/matmul_blocked4t.wgsl"),
    ),
    (
        "matmul/matmul_naive",
        include_str!("shaders/matmul/matmul_naive.wgsl"),
    ),
    (
        "matmul/matmul_tiled16",
        include_str!("shaders/matmul/matmul_tiled16.wgsl"),
    ),
    (
        "opt/accumulate",
        include_str!("shaders/opt/accumulate.wgsl"),
    ),
    ("opt/adam", include_str!("shaders/opt/adam.wgsl")),
    (
        "opt/norm_final",
        include_str!("shaders/opt/norm_final.wgsl"),
    ),
    (
        "opt/sumsq_part",
        include_str!("shaders/opt/sumsq_part.wgsl"),
    ),
    ("quant/q8_snap", include_str!("shaders/quant/q8_snap.wgsl")),
    ("quant/qlinear", include_str!("shaders/quant/qlinear.wgsl")),
];

/// The source of kernel `path` (`group/name`), if registered.
pub fn source(path: &str) -> Option<&'static str> {
    SOURCES.iter().find(|(n, _)| *n == path).map(|(_, s)| *s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_lists_exactly_the_shader_files_on_disk() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let mut on_disk = Vec::new();
        for group in std::fs::read_dir(&root).unwrap().flatten() {
            for f in std::fs::read_dir(group.path()).unwrap().flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                if let Some(stem) = name.strip_suffix(".wgsl") {
                    on_disk.push(format!("{}/{stem}", group.file_name().to_string_lossy()));
                }
            }
        }
        on_disk.sort();
        let registered: Vec<String> = SOURCES.iter().map(|(n, _)| n.to_string()).collect();
        assert_eq!(
            registered, on_disk,
            "registry and src/shaders/ differ: regenerate shaders.rs"
        );
        let mut dedup = registered.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), registered.len(), "names are unique");
    }

    #[test]
    fn every_registered_kernel_parses_and_validates_as_wgsl_with_the_prelude() {
        for (name, src) in SOURCES {
            let code = format!("{PRELUDE}\n{src}");
            let module = naga::front::wgsl::parse_str(&code)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            let flags = naga::valid::ValidationFlags::all();
            naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn lookup_finds_registered_kernels_and_refuses_others() {
        assert!(source("llama/rmsnorm").unwrap().contains("fn main"));
        assert!(source("llama/../rmsnorm").is_none());
        assert!(source("").is_none());
        assert!(PRELUDE.contains("fn gid"));
    }
}
