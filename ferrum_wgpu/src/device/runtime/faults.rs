use super::*;
use std::sync::atomic::AtomicUsize;

/// How many commands, readbacks and allocations may still succeed before every later one fails
/// (`usize::MAX`: never fail).
pub(crate) struct Faults {
    pub(crate) commands: AtomicUsize,
    pub(crate) reads: AtomicUsize,
    pub(crate) allocs: AtomicUsize,
}

impl Default for Faults {
    fn default() -> Self {
        Faults {
            commands: AtomicUsize::new(usize::MAX),
            reads: AtomicUsize::new(usize::MAX),
            allocs: AtomicUsize::new(usize::MAX),
        }
    }
}

impl Faults {
    /// Count one operation against `left`: `Err` once it is used up.
    pub(crate) fn tick(left: &AtomicUsize) -> Result<(), GpuError> {
        match left.load(Ordering::Relaxed) {
            usize::MAX => Ok(()),
            0 => Err(GpuError("injected fault".into())),
            n => {
                left.store(n - 1, Ordering::Relaxed);
                Ok(())
            }
        }
    }
}
