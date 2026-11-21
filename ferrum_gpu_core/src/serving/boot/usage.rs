use super::*;

/// The usage text.
pub const USAGE: &str = "usage: ferrum-gpu-server --model F.gguf [--host 127.0.0.1] [--port 8080] [--token T | FERRUM_TOKEN] \
[--allow-unauthenticated] [--max-batch N] [--max-chars N] [--max-body BYTES] [--max-tokens N] \
[--repetition-penalty X] [--device cpu|gpu|auto] [--gpu-adapter NAME] [--gpu-mem-mb N] [--threads N]";

/// A started server.
pub struct Booted {
    /// The running server (stop it with [`Running::stop`]).
    pub server: Running,
    /// Where it listens.
    pub addr: SocketAddr,
    /// Requests must carry a bearer token.
    pub has_token: bool,
}

/// Was help asked for?
pub fn wants_help(args: &[String]) -> bool {
    args.iter().any(|a| a == "-h" || a == "--help")
}

pub(super) fn say(out: &mut dyn Write, s: String) {
    // a closed stdout must not kill a server that is otherwise fine
    let _ = writeln!(out, "{s}");
}
