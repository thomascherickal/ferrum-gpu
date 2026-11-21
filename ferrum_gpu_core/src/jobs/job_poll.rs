use super::*;

impl Job {
    /// The lines from index `since` on, and the state.
    pub fn poll(&self, since: usize) -> Poll {
        let word = self.word();
        let i = lock(&self.inner);
        let from = since.max(i.base);
        let lines = i
            .lines
            .iter()
            .enumerate()
            .skip(from - i.base)
            .map(|(k, l)| (i.base + k, l.clone()))
            .collect();
        let exit = match &i.state {
            State::Ended(c) => Some(*c),
            State::Running => None,
        };
        Poll {
            state: word,
            exit,
            lines,
            next: i.base + i.lines.len(),
            last_step: i.last_step.clone(),
            last_val: i.last_val.clone(),
            last_eval: i.last_eval.clone(),
            seconds: i
                .finished
                .unwrap_or_else(Instant::now)
                .duration_since(self.started)
                .as_secs_f64(),
        }
    }

    /// Ask the job to stop. Cooperative: training and generation notice at the next step or token
    /// and end cleanly (a training run writes the model it has so far); a command without such a
    /// check (prepare, tokenize, the exports) runs to its end. False if the job had already finished.
    pub fn stop(&self) -> bool {
        if self.is_finished() {
            return false;
        }
        self.stopping.store(true, Ordering::SeqCst);
        self.cancel.store(true, Ordering::SeqCst);
        true
    }
}
