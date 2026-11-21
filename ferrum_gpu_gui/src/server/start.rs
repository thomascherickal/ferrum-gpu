use super::*;

/// Listen on `127.0.0.1:port` and route requests to `app`.
pub fn start(app: Arc<App>, port: u16) -> Result<Running, String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let handler: Handler = Arc::new(move |req| route(&app, req));
    let s = stop.clone();
    let thread = std::thread::spawn(move || serve(listener, handler, s, MAX_BODY));
    Ok(Running {
        addr,
        stop,
        thread: Some(thread),
    })
}
