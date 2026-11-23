use super::*;

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
    pub(super) server: Mutex<Option<Running>>,
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
