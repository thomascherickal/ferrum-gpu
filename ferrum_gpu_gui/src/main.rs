//! `ferrum_gpu_gui`: serve the ferrum-gpu GUI on loopback and wait.

use ferrum_gpu_gui::app::App;
use ferrum_gpu_gui::server::{parse_args, start, USAGE};
use std::io::Write;
use std::sync::Arc;

/// Print a status line. A closed stdout (`ferrum_gpu_gui | head -1`, a parent that stopped
/// reading) must not kill a server that is otherwise fine, which `println!` would (it panics on a
/// broken pipe).
macro_rules! say {
    ($($t:tt)*) => {
        let _ = writeln!(std::io::stdout(), $($t)*);
    };
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if opts.help {
        say!("{USAGE}");
        return;
    }
    let app = match App::new(opts.device, opts.adapter.as_deref(), opts.threads) {
        Ok(a) => Arc::new(a),
        Err(e) => {
            eprintln!("cannot start: {e}");
            std::process::exit(1);
        }
    };
    say!("device: {} ({})", app.backend.name(), app.backend.kind());
    if let Some(n) = &app.fallback_note {
        say!("note: {n}");
    }
    match start(app, opts.port) {
        Ok(server) => {
            say!("ferrum-gpu GUI: {}", server.url());
            say!("press Ctrl-C to stop");
            loop {
                std::thread::park();
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
