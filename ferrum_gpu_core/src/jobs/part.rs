use super::*;

impl Jobs {
    /// A job table whose jobs run the real commands.
    pub fn new() -> Jobs {
        Jobs::with_runner(Arc::new(|args, out| crate::commands::run(args, out)))
    }

    /// A job table with another command runner (tests use fakes).
    pub fn with_runner(run: Arc<RunFn>) -> Jobs {
        Jobs {
            next: AtomicU64::new(1),
            table: Mutex::new(BTreeMap::new()),
            run,
        }
    }

    /// Validate and start a job.
    pub fn start(&self, cmd: &str, args: Vec<String>) -> Result<Arc<Job>, String> {
        allow::validate(cmd, &args)?;
        let mut t = lock(&self.table);
        let running: Vec<_> = t.values().filter(|j| !j.is_finished()).collect();
        if running.len() >= MAX_RUNNING {
            return Err(format!(
                "{MAX_RUNNING} jobs are already running; wait or stop one"
            ));
        }
        // adapter enumeration is not safe from several threads at once, so the commands that open a
        // device take turns
        if allow::uses_device(cmd) {
            if let Some(other) = running.iter().find(|j| allow::uses_device(&j.cmd)) {
                return Err(format!(
                    "job {} ({}) is using the device; wait for it or stop it first",
                    other.id, other.cmd
                ));
            }
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let job = Arc::new(Job::new(id, cmd, args));
        runner::spawn(self.run.clone(), job.clone())?;
        t.insert(id, job.clone());
        // forget the oldest finished jobs beyond MAX_KEPT (there are always enough: at most
        // MAX_RUNNING jobs can be unfinished, and MAX_RUNNING is far below MAX_KEPT)
        let excess = t.len().saturating_sub(MAX_KEPT);
        let forget: Vec<u64> = t
            .iter()
            .filter(|(_, j)| j.is_finished())
            .map(|(k, _)| *k)
            .take(excess)
            .collect();
        for k in forget {
            t.remove(&k);
        }
        Ok(job)
    }

    /// A job by id.
    pub fn get(&self, id: u64) -> Option<Arc<Job>> {
        lock(&self.table).get(&id).cloned()
    }

    /// Every remembered job, oldest first.
    pub fn list(&self) -> Vec<Arc<Job>> {
        lock(&self.table).values().cloned().collect()
    }
}
