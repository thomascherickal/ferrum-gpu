use super::*;

/// An engine whose every step fails.
pub(super) struct Failing;
impl Engine for Failing {
    fn step(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: usize,
        _: &StepConfig,
    ) -> Result<StepStats, String> {
        Err("device exploded".into())
    }
    fn steps(&self) -> u64 {
        0
    }
    fn snapshot(&self) -> Result<Snapshot, String> {
        Err("no state".into())
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
}

/// A source that computes its tokens on demand, like a file read window by window.
pub(super) struct Lazy(pub(crate) usize);

impl TokenSource for Lazy {
    fn len(&self) -> usize {
        self.0
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        if start + len > self.0 {
            return Err("past the end".into());
        }
        Ok((start..start + len).map(|i| (i * 7) % 50).collect())
    }
}
