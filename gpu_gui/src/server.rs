//! Starting the server and parsing its command line.

use crate::api::{route, MAX_BODY};
use crate::app::{App, DeviceChoice};
use crate::http::{serve, Handler};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Port to listen on (`0` = any free port).
    pub port: u16,
    /// `--device`.
    pub device: DeviceChoice,
    /// `--gpu-adapter` name fragment.
    pub adapter: Option<String>,
    /// CPU matmul threads.
    pub threads: usize,
    /// `--help` was given.
    pub help: bool,
}

/// The usage text.
pub const USAGE: &str =
    "usage: gpu_gui [--port N] [--device cpu|gpu|auto] [--gpu-adapter NAME] [--threads N]\n\
    Serves the GUI on http://127.0.0.1:PORT/ (loopback only). Default: a free port, --device auto.";

/// Parse `args` (without the program name). Unknown flags and missing values are errors.
pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        port: 0,
        device: DeviceChoice::Auto,
        adapter: None,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        help: false,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--help" | "-h" => o.help = true,
            "--port" => {
                o.port = value("--port")?
                    .parse()
                    .map_err(|_| "--port must be 0..=65535".to_string())?
            }
            "--device" => o.device = DeviceChoice::parse(&value("--device")?)?,
            "--gpu-adapter" => o.adapter = Some(value("--gpu-adapter")?),
            "--threads" => {
                o.threads = value("--threads")?
                    .parse()
                    .ok()
                    .filter(|&n| n >= 1)
                    .ok_or("--threads must be at least 1")?
            }
            other => return Err(format!("unknown flag '{other}'\n{USAGE}")),
        }
    }
    Ok(o)
}

/// A running server.
pub struct Running {
    /// Where it listens.
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    /// The URL to open.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.addr.port())
    }

    /// Stop accepting connections and wait for the accept loop to end.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn flags_parse_and_unknown_or_incomplete_ones_are_refused() {
        let o = parse_args(&args(
            "--port 8080 --device gpu --gpu-adapter llvmpipe --threads 3",
        ))
        .unwrap();
        assert_eq!(
            (o.port, o.device, o.adapter.as_deref(), o.threads),
            (8080, DeviceChoice::Gpu, Some("llvmpipe"), 3)
        );
        let d = parse_args(&[]).unwrap();
        assert_eq!(
            (d.port, d.device, d.adapter, d.help),
            (0, DeviceChoice::Auto, None, false)
        );
        assert!(d.threads >= 1);
        assert!(parse_args(&args("--help")).unwrap().help);
        assert!(parse_args(&args("-h")).unwrap().help);
        for bad in [
            "--port",
            "--port x",
            "--port 70000",
            "--device",
            "--device tpu",
            "--threads 0",
            "--threads",
            "--gpu-adapter",
            "--nope",
            "stray",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad}");
        }
        assert!(parse_args(&args("--nope")).unwrap_err().contains("usage:"));
        for d in [DeviceChoice::Cpu, DeviceChoice::Gpu, DeviceChoice::Auto] {
            assert_eq!(DeviceChoice::parse(d.as_str()).unwrap(), d);
        }
    }
}
