//! A model wide enough for Q2_K (width 256, hidden 512) in the precisions added after Q4_0, trained
//! once per test process like the shared fixture.

use super::model::{ferrum_ok, s, SENTENCE};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// One trained model and its exports.
pub struct Wide {
    pub dir: PathBuf,
    pub f32: PathBuf,
    pub f16: PathBuf,
    pub fp4: PathBuf,
    pub q2: PathBuf,
}

impl Wide {
    /// The exported files with a label.
    pub fn all(&self) -> [(&'static str, &Path); 3] {
        [("f16", &self.f16), ("fp4", &self.fp4), ("q2", &self.q2)]
    }
}

/// The shared wide fixture (trained on first use).
pub fn wide() -> &'static Wide {
    static W: OnceLock<Wide> = OnceLock::new();
    W.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("ferrum_interop_wide_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, SENTENCE.repeat(60)).unwrap();
        let f32_ = dir.join("f32.gguf");
        ferrum_ok(&[
            "train",
            "--device",
            "cpu",
            "--context",
            "16",
            "--embed",
            "256",
            "--heads",
            "4",
            "--blocks",
            "1",
            "--hidden",
            "512",
            "--batch",
            "8",
            "--epochs",
            "40",
            "--vocab",
            "300",
            "--lr",
            "0.005",
            "--quant",
            "f32",
            "--data",
            s(&corpus),
            "--out",
            s(&f32_),
        ]);
        let export = |name: &str, quant: &str| {
            let out = dir.join(name);
            ferrum_ok(&[
                "export-gguf",
                "--model",
                s(&f32_),
                "--out",
                s(&out),
                "--quant",
                quant,
            ]);
            out
        };
        Wide {
            f16: export("f16.gguf", "f16"),
            fp4: export("fp4.gguf", "fp4"),
            q2: export("q2.gguf", "q2"),
            f32: f32_,
            dir,
        }
    })
}
