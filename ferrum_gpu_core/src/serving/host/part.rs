use super::*;

impl ServeSlot {
    /// An empty slot.
    pub fn new() -> ServeSlot {
        ServeSlot::default()
    }

    /// Start a server from the `ferrum-gpu-server` flags in `args`. Refused while one is running.
    pub fn start(&self, args: &[String]) -> Result<Status, String> {
        let mut h = lock(&self.hosted);
        if h.is_some() {
            return Err("a server is already running: stop it first".into());
        }
        lock(&self.lines).clear();
        let mut sink = Lines(self.lines.clone(), vec![]);
        match boot(args, &mut sink) {
            Ok(booted) => {
                *h = Some(Hosted {
                    booted,
                    since: Instant::now(),
                });
            }
            Err(e) => {
                lock(&self.lines).push(format!("could not start: {e}"));
                return Err(e);
            }
        }
        drop(h);
        Ok(self.status())
    }

    /// Stop the server. False if none was running.
    pub fn stop(&self) -> bool {
        let h = lock(&self.hosted).take();
        match h {
            Some(h) => {
                h.booted.server.stop();
                lock(&self.lines).push("server stopped".into());
                true
            }
            None => false,
        }
    }

    /// The current status.
    pub fn status(&self) -> Status {
        let h = lock(&self.hosted);
        Status {
            running: h.is_some(),
            addr: h.as_ref().map(|h| h.booted.addr.to_string()),
            has_token: h.as_ref().is_some_and(|h| h.booted.has_token),
            seconds: h.as_ref().map(|h| h.since.elapsed().as_secs_f64()),
            lines: lock(&self.lines).clone(),
        }
    }
}
