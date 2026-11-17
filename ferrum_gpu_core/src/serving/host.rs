//! A slot that holds at most one running server, for a front end that starts and stops it from a
//! window (the Studio's Serve tab). It starts the server exactly as the command line does
//! ([`boot`](super::boot::boot)) and keeps the status lines it printed.

use super::boot::{boot, Booted};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// Most status lines kept.
pub const MAX_LINES: usize = 200;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the slot holds, as seen from outside.
#[derive(Clone, Debug, PartialEq)]
pub struct Status {
    /// A server is running.
    pub running: bool,
    /// Its address (`127.0.0.1:8080`), if running.
    pub addr: Option<String>,
    /// Requests need a bearer token.
    pub has_token: bool,
    /// Seconds since it started, if running.
    pub seconds: Option<f64>,
    /// The status lines the server printed when it started (and why it did not, if it failed).
    pub lines: Vec<String>,
}

struct Hosted {
    booted: Booted,
    since: Instant,
}

/// The slot.
#[derive(Default)]
pub struct ServeSlot {
    hosted: Mutex<Option<Hosted>>,
    lines: Arc<Mutex<Vec<String>>>,
}

/// A writer that keeps the lines written to it (bounded).
struct Lines(Arc<Mutex<Vec<String>>>, Vec<u8>);

impl Write for Lines {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        for &b in data {
            if b == b'\n' {
                let s = String::from_utf8_lossy(&self.1).into_owned();
                self.1.clear();
                let mut v = lock(&self.0);
                v.push(s);
                if v.len() > MAX_LINES {
                    v.remove(0);
                }
            } else {
                self.1.push(b);
            }
        }
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

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

#[cfg(test)]
mod tests;
