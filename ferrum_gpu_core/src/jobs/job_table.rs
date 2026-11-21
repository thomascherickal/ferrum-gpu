use super::*;

/// All jobs.
pub struct Jobs {
    pub(crate) next: AtomicU64,
    pub(crate) table: Mutex<BTreeMap<u64, Arc<Job>>>,
    pub(crate) run: Arc<RunFn>,
}

impl Default for Jobs {
    fn default() -> Self {
        Jobs::new()
    }
}
