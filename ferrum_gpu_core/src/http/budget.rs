use super::*;

/// Default most connections served at once; more are refused with 503 and `Retry-After`.
pub const MAX_CONNECTIONS: usize = 128;

/// The total time a client gets to send one request, however it paces its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// For the whole head; also the longest single read on a socket.
    pub head: Duration,
    /// For the body: `head` plus this much per started MiB of `Content-Length`.
    pub per_mib: Duration,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            head: Duration::from_secs(10),
            per_mib: Duration::from_secs(10),
        }
    }
}

impl Budget {
    /// The time allowed for a body of `len` bytes.
    pub fn body(&self, len: usize) -> Duration {
        let mibs = u32::try_from(len.div_ceil(1 << 20)).unwrap_or(u32::MAX);
        self.head.saturating_add(self.per_mib.saturating_mul(mibs))
    }
}

/// How the server treats each connection.
#[derive(Clone, Copy, Debug)]
pub struct ServeOptions {
    /// Largest request body accepted (413 past it).
    pub max_body: usize,
    /// Apply the Host/Origin check of [`guard`] before the handler.
    pub loopback_guard: bool,
    /// Time allowed to read a request (408 past it).
    pub budget: Budget,
    /// Most connection threads at once (503 past it).
    pub max_connections: usize,
}

impl ServeOptions {
    /// The defaults: [`Budget::default`] and [`MAX_CONNECTIONS`].
    pub fn new(max_body: usize, loopback_guard: bool) -> Self {
        ServeOptions {
            max_body,
            loopback_guard,
            budget: Budget::default(),
            max_connections: MAX_CONNECTIONS,
        }
    }
}

/// A reader that fails with `TimedOut` once `allowed` has passed since `start` on `now`'s clock.
pub(super) struct Timed<'a> {
    pub(super) inner: &'a mut dyn Read,
    pub(super) now: &'a dyn Fn() -> Instant,
    pub(super) start: Instant,
    pub(super) allowed: Duration,
}

impl Read for Timed<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if (self.now)().saturating_duration_since(self.start) >= self.allowed {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        self.inner.read(buf)
    }
}
