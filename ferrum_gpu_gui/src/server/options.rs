use super::*;

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
    "usage: ferrum_gpu_gui [--port N] [--device cpu|gpu|auto] [--gpu-adapter NAME] [--threads N]\n\
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
    pub(super) stop: Arc<AtomicBool>,
    pub(super) thread: JoinHandle<()>,
}

impl Running {
    /// The URL to open.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.addr.port())
    }

    /// Stop accepting connections and wait for the accept loop to end.
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }
}
