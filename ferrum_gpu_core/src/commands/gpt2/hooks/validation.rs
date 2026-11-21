use super::*;

/// Held-out tokens and the early-stopping rule.
pub(in super::super) struct Validation {
    /// Tokens held out from the end of the corpus.
    pub(in super::super) tokens: Vec<usize>,
    /// Stop after this many epochs without improvement (`0` = never stop early).
    pub(in super::super) patience: usize,
}

/// The step hook of `train`.
pub(in super::super) struct Run<'a> {
    pub(in super::super) out: &'a mut dyn Write,
    pub(in super::super) verbose: bool,
    pub(in super::super) csv: Option<Box<dyn Write>>,
    pub(in super::super) started: Instant,
    pub(in super::super) max_seconds: Option<f64>,
    pub(in super::super) stop_file: PathBuf,
    pub(in super::super) checkpoint: Option<PathBuf>,
    pub(in super::super) sched: Option<Schedule>,
    pub(in super::super) lr: f32,
    pub(in super::super) tokens_per_step: usize,
    pub(in super::super) steps_per_epoch: u64,
    pub(in super::super) total: u64,
    pub(in super::super) val: Option<Validation>,
    /// Best validation loss so far and the weights that gave it.
    pub(in super::super) best: Option<(f64, Gpt2)>,
    pub(in super::super) bad_epochs: usize,
    /// The most recent weights seen at a checkpoint.
    pub(in super::super) last: Option<Gpt2>,
    /// Why the run stopped early, if it did.
    pub(in super::super) reason: Option<String>,
    /// A write failed; reported after the run.
    pub(in super::super) error: Option<String>,
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
