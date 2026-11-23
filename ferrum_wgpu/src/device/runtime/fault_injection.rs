use super::*;

impl Gpu {
    /// Fault injection: let `n` more commands (dispatches and copies) succeed, then fail every
    /// later one with "injected fault". `usize::MAX` turns injection off. Used to drive the error
    /// branch of every `?` in code that issues many commands.
    #[cfg(any(test, feature = "fault-injection"))]
    pub fn fail_after(&self, n: usize) {
        self.faults.commands.store(n, Ordering::Relaxed);
    }

    /// Fault injection: let `n` more readbacks succeed, then fail every later one.
    #[cfg(any(test, feature = "fault-injection"))]
    pub fn fail_reads_after(&self, n: usize) {
        self.faults.reads.store(n, Ordering::Relaxed);
    }

    /// Fault injection: let `n` more buffer allocations succeed, then fail every later one.
    #[cfg(any(test, feature = "fault-injection"))]
    pub fn fail_allocs_after(&self, n: usize) {
        self.faults.allocs.store(n, Ordering::Relaxed);
    }

    #[cfg(any(test, feature = "fault-injection"))]
    pub(crate) fn injected_alloc_fault(&self) -> Result<(), GpuError> {
        Faults::tick(&self.faults.allocs)
    }

    #[cfg(any(test, feature = "fault-injection"))]
    pub(crate) fn injected_fault(&self) -> Result<(), GpuError> {
        Faults::tick(&self.faults.commands)
    }

    #[cfg(any(test, feature = "fault-injection"))]
    pub(crate) fn injected_read_fault(&self) -> Result<(), GpuError> {
        Faults::tick(&self.faults.reads)
    }

    /// Test helper: run `f` repeatedly, failing command 0, then 1, then 2, ... until it
    /// succeeds. Every failed run must report the injected fault and leave no batch open.
    /// Returns how many commands `f` issues, and turns injection off again.
    #[cfg(any(test, feature = "fault-injection"))]
    pub fn sweep_faults(&self, mut f: impl FnMut() -> Result<(), GpuError>) -> usize {
        self.sweep_faults_dyn(&mut f)
    }

    #[cfg(any(test, feature = "fault-injection"))]
    fn sweep_faults_dyn(&self, f: &mut dyn FnMut() -> Result<(), GpuError>) -> usize {
        for k in 0..100_000 {
            self.fail_after(k);
            match f() {
                Err(e) => {
                    assert!(e.to_string().contains("injected fault"), "command {k}: {e}");
                    assert!(
                        !self.batching(),
                        "a batch was left open after a fault at command {k}"
                    );
                }
                Ok(()) => {
                    self.fail_after(usize::MAX);
                    return k;
                }
            }
        }
        panic!("the operation never succeeded");
    }

    /// Start recording every following dispatch and copy into one command encoder (one submit
    /// instead of one per kernel). Reads are refused until [`Gpu::end_batch`].
    pub fn begin_batch(&self) -> Result<(), GpuError> {
        self.check()?;
        let mut batch = self.batch.lock().unwrap_or_else(|e| e.into_inner());
        if batch.is_some() {
            return Err(GpuError("a batch is already open".into()));
        }
        *batch = Some(self.device.create_command_encoder(&Default::default()));
        Ok(())
    }

    /// Close the batch. With `submit` the recorded work runs; without it the work is dropped.
    /// Errors raised while recording are reported here.
    pub fn end_batch(&self, submit: bool) -> Result<(), GpuError> {
        let enc = self
            .batch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| GpuError("no batch is open".into()))?;
        if submit {
            self.queue.submit([enc.finish()]);
        }
        self.check()
    }
}
