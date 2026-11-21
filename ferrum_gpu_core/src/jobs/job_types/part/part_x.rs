use super::*;

impl Job {
    pub(crate) fn new(id: u64, cmd: &str, args: Vec<String>) -> Job {
        Job {
            id,
            cmd: cmd.into(),
            args,
            started: Instant::now(),
            inner: Mutex::new(Inner {
                lines: VecDeque::new(),
                base: 0,
                state: State::Running,
                last_step: None,
                last_val: None,
                last_eval: None,
                finished: None,
            }),
            stopping: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Append an output line (parsing progress), dropping the oldest beyond [`MAX_LINES`].
    pub(crate) fn push(&self, err: bool, text: String) {
        let ev = progress::parse_line(&text);
        let mut i = lock(&self.inner);
        match &ev {
            Some(e @ Event::Step { .. }) => i.last_step = Some(e.clone()),
            Some(e @ Event::Val { .. }) => i.last_val = Some(e.clone()),
            Some(e @ Event::Eval { .. }) => i.last_eval = Some(e.clone()),
            None => {}
        }
        i.lines.push_back(Line { err, text, ev });
        if i.lines.len() > MAX_LINES {
            i.lines.pop_front();
            i.base += 1;
        }
    }

    pub(crate) fn finish(&self, state: State) {
        let mut i = lock(&self.inner);
        i.state = state;
        i.finished = Some(Instant::now());
    }

    /// The status word: `running`, `stopping`, `done`, `stopped` or `error`.
    pub fn word(&self) -> &'static str {
        let stopping = self.stopping.load(Ordering::SeqCst);
        match &lock(&self.inner).state {
            State::Running if stopping => "stopping",
            State::Running => "running",
            State::Ended(_) if stopping => "stopped",
            State::Ended(0) => "done",
            State::Ended(_) => "error",
        }
    }

    /// True once the command has ended.
    pub fn is_finished(&self) -> bool {
        lock(&self.inner).state != State::Running
    }
}
