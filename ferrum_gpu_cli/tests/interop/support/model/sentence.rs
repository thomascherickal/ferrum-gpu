use super::*;

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

pub fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// A static cannot be dropped, so a finished run leaves its fixture directory behind; the next run
/// removes the ones older than an hour (never a newer one: another run may be using it).
pub(super) fn sweep_old_fixtures() {
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
