#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! Ferrum GPU Studio: the desktop app.
//!
//! The app is a native window (Tauri 2) around the web GUI that ships with `ferrum_gpu_gui`. It starts
//! that GUI's loopback server **inside this process** on a free port, with the engine linked in (jobs
//! run the `ferrum-gpu` commands on threads here; there is no child process and no binary to find),
//! and points the window at it. Everything the page does goes through that server, whose Host/Origin
//! guard and Content-Security-Policy apply unchanged; the window itself is granted no native commands.
//!
//! The same page also runs in an ordinary browser (`ferrum_gpu_gui`), which is how it is tested in a
//! real browser; this crate adds the window, the lifetime of the server, and the packaging.

use ferrum_gpu_gui::app::App;
use ferrum_gpu_gui::server::{parse_args, start, Running, USAGE};
use std::io::Write;
use std::sync::{Arc, Mutex};
use tauri::{Manager, RunEvent, Runtime, WebviewUrl, WebviewWindowBuilder};

/// The window title.
pub const TITLE: &str = "Ferrum GPU Studio";

/// What `run` does with a command line.
#[derive(Debug, PartialEq)]
pub enum Launch {
    /// Print this text and exit (`--help`).
    Help(&'static str),
    /// Start the app with these flags.
    Start(Vec<String>),
}

/// Decide what to do with the command line (without the program name).
pub fn launch(args: &[String]) -> Result<Launch, String> {
    let o = parse_args(args)?;
    Ok(if o.help {
        Launch::Help(USAGE)
    } else {
        Launch::Start(args.to_vec())
    })
}

/// Open the device and start the loopback server for `args` (the GUI's flags: `--device`,
/// `--gpu-adapter`, `--threads`, `--port`).
pub fn start_server(args: &[String]) -> Result<Running, String> {
    let o = parse_args(args)?;
    let app = Arc::new(App::new(o.device, o.adapter.as_deref(), o.threads)?);
    start(app, o.port)
}

/// The server, held as app state so it lives exactly as long as the app.
pub struct Studio {
    server: Mutex<Option<Running>>,
    /// The page the window shows.
    pub url: String,
}

impl Studio {
    /// Wrap a running server.
    pub fn new(server: Running) -> Studio {
        let url = server.url();
        Studio {
            server: Mutex::new(Some(server)),
            url,
        }
    }

    /// Is the server still running?
    pub fn running(&self) -> bool {
        self.server
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Stop the server (a second call does nothing).
    pub fn stop(&self) {
        let s = self.server.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(s) = s {
            s.stop();
        }
    }
}

/// Register the server as state and create the main window on it. `run` and the tests both build the
/// app through this, so the tests exercise the real setup.
pub fn configure<R: Runtime>(builder: tauri::Builder<R>, server: Running) -> tauri::Builder<R> {
    let studio = Studio::new(server);
    let url = studio.url.clone();
    builder.manage(studio).setup(move |app| {
        WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
            .title(TITLE)
            .inner_size(1320.0, 880.0)
            .min_inner_size(920.0, 640.0)
            .build()?;
        Ok(())
    })
}

/// Start the app: parse the flags, open the device, serve, open the window, and stop the server when
/// the app exits.
pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match launch(&args) {
        Ok(Launch::Help(text)) => {
            println!("{text}");
            return;
        }
        Ok(Launch::Start(a)) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let server = match start_server(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot start: {e}");
            std::process::exit(1);
        }
    };
    // a closed stdout (a launcher that stopped reading) must not kill the app, as `println!` would
    let _ = writeln!(std::io::stdout(), "Ferrum GPU Studio: {}", server.url());
    configure(tauri::Builder::default(), server)
        .build(tauri::generate_context!())
        .expect("cannot build the Ferrum GPU Studio window")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<Studio>().stop();
            }
        });
}

#[cfg(test)]
mod tests;
