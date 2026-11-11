//! `ferrum_gpu_gui`: serve the ferrum-gpu GUI on loopback and wait.

use ferrum_gpu_gui::app::App;
use ferrum_gpu_gui::server::{parse_args, start, USAGE};
use std::sync::Arc;

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
        println!("{USAGE}");
        return;
    }
    let app = match App::new(opts.device, opts.adapter.as_deref(), opts.threads) {
        Ok(a) => Arc::new(a),
        Err(e) => {
            eprintln!("cannot start: {e}");
            std::process::exit(1);
        }
    };
    println!("device: {} ({})", app.backend.name(), app.backend.kind());
    if let Some(n) = &app.fallback_note {
        println!("note: {n}");
    }
    match start(app, opts.port) {
        Ok(server) => {
            println!("ferrum-gpu GUI: {}", server.url());
            println!("press Ctrl-C to stop");
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
