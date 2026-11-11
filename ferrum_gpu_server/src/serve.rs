//! Start the server around any [`Decoder`]: bind, optional bearer token, in-flight cap.

use crate::auth;
use crate::decoder::Decoder;
use crate::limits::{Limits, MAX_INFLIGHT};
use crate::routes::route;
use ferrum_gpu_gui::http::{serve_with, Handler, Response};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

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
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "127.0.0.1".into(),
            port: 0,
            token: None,
            limits: Limits::default(),
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
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
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
struct Slot<'a>(&'a AtomicUsize);

impl<'a> Slot<'a> {
    fn take(n: &'a AtomicUsize) -> Option<Slot<'a>> {
        if n.fetch_add(1, Ordering::SeqCst) < MAX_INFLIGHT {
            Some(Slot(n))
        } else {
            n.fetch_sub(1, Ordering::SeqCst);
            None
        }
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Listen as `cfg` says.
pub fn start(d: Arc<dyn Decoder>, cfg: &Config) -> Result<Running, String> {
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .map_err(|e| format!("cannot listen on {}:{}: {e}", cfg.host, cfg.port))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let (token, limits, inflight) = (cfg.token.clone(), cfg.limits.clone(), AtomicUsize::new(0));
    let handler: Handler = Arc::new(move |req| {
        if let Some(denied) = token.as_deref().and_then(|t| auth::check(req, t)) {
            return denied;
        }
        let _slot = if req.method == "POST" {
            match Slot::take(&inflight) {
                Some(s) => Some(s),
                None => return Response::error(503, "too many requests in flight"),
            }
        } else {
            None
        };
        route(d.as_ref(), &limits, req)
    });
    let (s, max_body, guard) = (stop.clone(), cfg.limits.max_body, is_loopback(&cfg.host));
    let thread = std::thread::spawn(move || serve_with(listener, handler, s, max_body, guard));
    Ok(Running {
        addr,
        stop,
        thread: Some(thread),
    })
}

#[cfg(test)]
mod tests;
