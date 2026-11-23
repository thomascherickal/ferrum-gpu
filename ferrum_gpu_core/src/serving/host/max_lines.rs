use super::*;

/// Most status lines kept.
pub const MAX_LINES: usize = 200;

pub(super) use crate::sync::lock;

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

pub(super) struct Hosted {
    pub(super) booted: Booted,
    pub(super) since: Instant,
}

/// The slot.
#[derive(Default)]
pub struct ServeSlot {
    pub(super) hosted: Mutex<Option<Hosted>>,
    pub(super) lines: Arc<Mutex<Vec<String>>>,
}

/// A writer that keeps the lines written to it (bounded).
pub(super) struct Lines(pub(super) Arc<Mutex<Vec<String>>>, pub(super) Vec<u8>);

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
