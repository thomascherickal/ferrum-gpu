use super::*;

impl Gpu {
    /// Test hook: let `n` more commands (dispatches and copies) succeed, then fail every later
    /// one with "injected fault". `usize::MAX` turns injection off. Used to drive the error
    /// branch of every `?` in code that issues many commands.
    #[cfg(test)]
    pub(crate) fn fail_after(&self, n: usize) {
        self.fault_after.store(n, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn injected_fault(&self) -> Result<(), GpuError> {
        match self.fault_after.load(Ordering::Relaxed) {
            usize::MAX => Ok(()),
            0 => Err(GpuError("injected fault".into())),
            n => {
                self.fault_after.store(n - 1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    /// Test helper: run `f` repeatedly, failing command 0, then 1, then 2, ... until it
    /// succeeds. Every failed run must report the injected fault and leave no batch open.
    /// Returns how many commands `f` issues, and turns injection off again.
    #[cfg(test)]
    pub(crate) fn sweep_faults(&self, mut f: impl FnMut() -> Result<(), GpuError>) -> usize {
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
