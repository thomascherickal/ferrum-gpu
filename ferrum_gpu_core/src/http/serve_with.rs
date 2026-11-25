use super::*;

/// Like [`serve`]; with `loopback_guard` false the Host/Origin check is skipped, for a server
/// that listens beyond loopback and authenticates every request itself.
pub fn serve_with(
    listener: TcpListener,
    handler: Handler,
    stop: Arc<AtomicBool>,
    max_body: usize,
    loopback_guard: bool,
) {
    serve_opts(
        listener,
        handler,
        stop,
        ServeOptions::new(max_body, loopback_guard),
    );
}

/// Like [`serve_with`], with every limit in `opts`. A connection past `opts.max_connections` gets
/// a 503 with `Retry-After: 1` from the accept loop itself and is closed; no thread is started.
pub fn serve_opts(
    listener: TcpListener,
    handler: Handler,
    stop: Arc<AtomicBool>,
    opts: ServeOptions,
) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    listener.set_nonblocking(true).ok();
    let open = Arc::new(AtomicUsize::new(0));
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            // only this loop adds to `open`, so the check cannot race past the cap
            Ok((stream, _)) if open.load(Ordering::SeqCst) >= opts.max_connections => {
                refuse(stream)
            }
            Ok((stream, _)) => {
                open.fetch_add(1, Ordering::SeqCst);
                let (h, slot) = (handler.clone(), Open(open.clone()));
                std::thread::spawn(move || {
                    let _slot = slot;
                    handle(stream, &*h, port, opts)
                });
            }
            // no connection waiting (WouldBlock) or a transient accept error: look again shortly
            Err(_) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

/// One running connection thread; frees its place in the count when it ends.
struct Open(Arc<AtomicUsize>);

impl Drop for Open {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn handle(
    mut stream: TcpStream,
    handler: &(dyn Fn(&Request) -> Response + Send + Sync),
    port: u16,
    opts: ServeOptions,
) {
    stream.set_nonblocking(false).ok();
    let per_read = opts.budget.head.max(Duration::from_millis(1));
    stream.set_read_timeout(Some(per_read)).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
    let resp = match read_request_within(&mut stream, opts.max_body, opts.budget, &Instant::now) {
        Ok(req) => opts
            .loopback_guard
            .then(|| guard(&req, port))
            .flatten()
            .unwrap_or_else(|| handle_watched(&stream, &req, handler)),
        Err(e) => Response::error(e.status, &e.message),
    };
    let _ = write_response(&mut stream, &resp);
    close_gracefully(&mut stream);
}
