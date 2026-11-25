use super::*;
use std::io::Cursor;

fn parse(s: &str) -> Result<Request, HttpError> {
    read_request(&mut Cursor::new(s.as_bytes().to_vec()), 1024)
}

fn status(s: &str) -> u16 {
    parse(s).unwrap_err().status
}

/// Serve `work`, which runs on the request thread; returns the port and a stop switch.
fn serve_work(
    work: impl Fn(&Request) -> Response + Send + Sync + 'static,
) -> (u16, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    serve_work_with(ServeOptions::new(1024, false), work)
}

/// [`serve_work`] with every limit in `opts`.
fn serve_work_with(
    opts: ServeOptions,
    work: impl Fn(&Request) -> Response + Send + Sync + 'static,
) -> (u16, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let (s, h): (_, Handler) = (stop.clone(), Arc::new(work));
    let t = std::thread::spawn(move || serve_opts(listener, h, s, opts));
    (port, stop, t)
}

mod a_client_that_disconnects_mid_request_cancels_the_work;
mod a_dribbling_client_is_cut_off_at_the_budget;
mod a_request_head_that_is_not_utf8_is_refused;
mod a_stalled_client_times_out_with_408;
mod parses_a_get_with_query_and_headers;
mod past_the_connection_cap_clients_are_refused_and_capacity_returns;
mod the_close_watcher_tells_a_close_a_reset_and_silence_apart;
mod the_guard_accepts_only_this_loopback_server;
