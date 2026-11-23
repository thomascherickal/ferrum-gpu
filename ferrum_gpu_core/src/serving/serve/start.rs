use super::*;

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Listen as `cfg` says.
pub fn start(d: Arc<dyn Decoder>, cfg: &Config) -> Result<Running, String> {
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .map_err(|e| format!("cannot listen on {}:{}: {e}", cfg.host, cfg.port))?;
    let addr = crate::error_text::text(listener.local_addr())?;
    let stop = Arc::new(AtomicBool::new(false));
    let (token, limits, inflight) = (cfg.token.clone(), cfg.limits.clone(), AtomicUsize::new(0));
    let cap = cfg.max_inflight;
    let handler: Handler = Arc::new(move |req| {
        if let Some(denied) = token.as_deref().and_then(|t| auth::check(req, t)) {
            return denied;
        }
        let _slot = if req.method == "POST" {
            match Slot::take(&inflight, cap) {
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
