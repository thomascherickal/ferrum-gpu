use super::*;

/// The shared fixture (trained on first use; later callers wait for it).
pub(crate) fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        sweep_old_fixtures();
        let dir = std::env::temp_dir().join(format!("ferrum_interop_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, SENTENCE.repeat(60)).unwrap();
        let (f32_, q8, q4) = (
            dir.join("f32.gguf"),
            dir.join("q8.gguf"),
            dir.join("q4.gguf"),
        );
        let tok = dir.join("tok.gguf");
        ferrum_ok(&[
            "train",
            "--device",
            "cpu",
            "--context",
            "16",
            "--embed",
            "32",
            "--heads",
            "2",
            "--blocks",
            "2",
            "--hidden",
            "64",
            "--batch",
            "8",
            "--epochs",
            "40",
            "--vocab",
            "300",
            "--lr",
            "0.01",
            "--quant",
            "f32",
            "--data",
            s(&corpus),
            "--out",
            s(&f32_),
            "--tokenizer",
            s(&tok),
        ]);
        ferrum_ok(&[
            "export-gguf",
            "--model",
            s(&f32_),
            "--out",
            s(&q8),
            "--quant",
            "int8",
        ]);
        ferrum_ok(&[
            "export-gguf",
            "--model",
            s(&f32_),
            "--out",
            s(&q4),
            "--quant",
            "int4",
        ]);
        Fixture {
            dir,
            corpus,
            f32: f32_,
            q8,
            q4,
        }
    })
}
