use super::*;

/// Held-out tokens and the early-stopping rule.
pub struct Validation {
    /// Tokens held out from the end of the corpus.
    pub tokens: Vec<usize>,
    /// Stop after this many epochs without improvement (`0` = never stop early).
    pub patience: usize,
}

/// The step hook of `train`.
pub struct Run<'a> {
    pub out: &'a mut dyn Write,
    pub verbose: bool,
    pub csv: Option<Box<dyn Write>>,
    pub started: Instant,
    pub max_seconds: Option<f64>,
    pub stop_file: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub sched: Option<Schedule>,
    pub lr: f32,
    pub tokens_per_step: usize,
    pub steps_per_epoch: u64,
    pub total: u64,
    pub val: Option<Validation>,
    /// Best validation loss so far and the weights that gave it.
    pub best: Option<(f64, Gpt2)>,
    pub bad_epochs: usize,
    /// The most recent weights seen at a checkpoint.
    pub last: Option<Gpt2>,
    /// Why the run stopped early, if it did.
    pub reason: Option<String>,
    /// A write failed; reported after the run.
    pub error: Option<String>,
}

impl Run<'_> {
    pub(super) fn note(&mut self, s: String) {
        if let Err(e) = writeln!(self.out, "{s}") {
            self.error.get_or_insert(e.to_string());
        }
    }

    pub(super) fn lr_at(&self, step: u64) -> f32 {
        self.sched.as_ref().map_or(self.lr, |s| s.lr_at(step))
    }
}
