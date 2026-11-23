use super::*;

pub(crate) const FAST: &str =
    "--device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4";

pub(crate) fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

pub(crate) fn ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&words(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

pub(crate) fn err(s: &str) -> String {
    run(&words(s), &mut Vec::new()).unwrap_err()
}

/// A scratch directory holding `c.txt`, a short repetitive corpus.
pub(crate) fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("ferrum_errp_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let c = d.join("c.txt");
    std::fs::write(&c, "the cat sat on the mat. ".repeat(60)).unwrap();
    (d, c)
}

/// Train a tiny model on `corpus` with a BPE vocabulary of `vocab`; returns its path and the path
/// of the tokenizer file written next to it.
pub(crate) fn trained(dir: &Path, corpus: &Path, vocab: usize, tag: &str) -> (PathBuf, PathBuf) {
    let (m, t) = (
        dir.join(format!("{tag}.gguf")),
        dir.join(format!("{tag}.tok.gguf")),
    );
    ok(&format!(
        "train {FAST} --epochs 2 --vocab {vocab} --data {} --out {} --tokenizer {}",
        corpus.display(),
        m.display(),
        t.display()
    ));
    (m, t)
}

/// A writer that accepts `budget` bytes and then fails every write, like a closed pipe.
pub(super) struct FailAfter {
    pub budget: usize,
}

impl Write for FailAfter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.budget == 0 {
            return Err(std::io::Error::other("output closed"));
        }
        let n = buf.len().min(self.budget);
        self.budget -= n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `cmd` writing to a [`FailAfter`] writer.
pub(super) fn run_failing(cmd: &str, budget: usize) -> Result<(), String> {
    run(&words(cmd), &mut FailAfter { budget })
}

/// A writer that succeeds for `n` writes and then fails every write: the `n`-th place a command
/// writes its output is the first one to fail.
pub(super) struct FailNth {
    pub(super) n: usize,
    pub(super) calls: usize,
}
