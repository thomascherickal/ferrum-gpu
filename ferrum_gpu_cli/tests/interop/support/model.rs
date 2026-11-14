//! The small models every interop test uses, trained once per test process.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// What the models are trained on: one sentence, repeated, so greedy text is unambiguous.
pub const SENTENCE: &str = "the cat sat on the mat. ";
/// A prompt every model continues the same way.
pub const PROMPT: &str = "the cat";
/// Tokens generated in the comparisons.
pub const NEW_TOKENS: usize = 10;

/// Run `ferrum-gpu` with `args`.
pub fn ferrum(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"))
        .args(args)
        .output()
        .expect("run ferrum-gpu")
}

/// `ferrum-gpu` that must succeed; its stdout.
pub fn ferrum_ok(args: &[&str]) -> String {
    let o = ferrum(args);
    assert!(
        o.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// The trained models: one directory, three precisions, one tokenizer.
pub struct Fixture {
    pub dir: PathBuf,
    pub corpus: PathBuf,
    pub f32: PathBuf,
    pub q8: PathBuf,
    pub q4: PathBuf,
}

impl Fixture {
    /// The three model files with a label.
    pub fn all(&self) -> [(&'static str, &Path); 3] {
        [("f32", &self.f32), ("q8_0", &self.q8), ("q4_0", &self.q4)]
    }
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// A static cannot be dropped, so a finished run leaves its fixture directory behind; the next run
/// removes the ones older than an hour (never a newer one: another run may be using it).
fn sweep_old_fixtures() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for e in entries.flatten() {
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > std::time::Duration::from_secs(3600));
        if old
            && e.file_name()
                .to_string_lossy()
                .starts_with("ferrum_interop_")
        {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// The shared fixture (trained on first use; later callers wait for it).
pub fn fixture() -> &'static Fixture {
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

/// Our greedy continuation of [`PROMPT`] with `model` (the prompt, then the text).
pub fn ours(model: &Path) -> String {
    ferrum_ok(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        s(model),
        "--prompt",
        PROMPT,
        "--max-new",
        &NEW_TOKENS.to_string(),
    ])
    .trim_end()
    .to_string()
}
