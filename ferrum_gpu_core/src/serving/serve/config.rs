use super::*;

/// Where and how to listen.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Address to bind (`127.0.0.1` unless you mean to expose the server).
    pub host: String,
    /// Port (`0` = any free port).
    pub port: u16,
    /// Bearer token every request but `GET /health` must carry.
    pub token: Option<String>,
    /// Request limits.
    pub limits: Limits,
    /// Most POST requests held at once; more get an immediate 503.
    pub max_inflight: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "127.0.0.1".into(),
            port: 0,
            token: None,
            limits: Limits::default(),
            max_inflight: MAX_INFLIGHT,
        }
    }
}

/// Is `host` a loopback address?
pub fn is_loopback(host: &str) -> bool {
    host == "localhost" || host.parse::<IpAddr>().is_ok_and(|a| a.is_loopback())
}

/// Refuse to listen beyond loopback without a token, unless `allow_unauthenticated`.
pub fn check_exposure(
    host: &str,
    token: Option<&str>,
    allow_unauthenticated: bool,
) -> Result<(), String> {
    if is_loopback(host) || token.is_some() || allow_unauthenticated {
        return Ok(());
    }
    Err(format!(
        "refusing to listen on {host}: it is not a loopback address and no token is set; pass --token (or set FERRUM_TOKEN), or --allow-unauthenticated if you really mean it"
    ))
}

/// A running server.
pub struct Running {
    /// Where it listens.
    pub addr: SocketAddr,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) thread: Option<JoinHandle<()>>,
}

impl Running {
    /// Stop accepting connections and wait for the accept loop to end.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Holds one in-flight slot until dropped.
pub(crate) struct Slot<'a>(pub(crate) &'a AtomicUsize);

impl<'a> Slot<'a> {
    pub(super) fn take(n: &'a AtomicUsize, cap: usize) -> Option<Slot<'a>> {
        if n.fetch_add(1, Ordering::SeqCst) < cap {
            Some(Slot(n))
        } else {
            n.fetch_sub(1, Ordering::SeqCst);
            None
        }
    }
}
