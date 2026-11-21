use super::*;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prepare_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn opts(input: &Path, out: &Path) -> PrepareOptions {
    PrepareOptions {
        inputs: vec![input.to_string_lossy().into()],
        out: out.to_string_lossy().into(),
        doc_sep: "<|endoftext|>".into(),
        lossy: false,
        nfkc: false,
        lowercase: false,
        strip_gutenberg: false,
        min_chars: 1,
        max_chars: 0,
        dedup: false,
        val_frac: 0.0,
        test_frac: 0.0,
        seed: 42,
        args: vec![],
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/prepare")
        .join(name)
}

mod a_run_cannot_read_its_own_outputs;
mod invalid_utf8_fails_with_the_offset_or_is_repaired_with_lossy;
mod matches_ferrum_cpus_prepare_on_the_golden_corpus;
