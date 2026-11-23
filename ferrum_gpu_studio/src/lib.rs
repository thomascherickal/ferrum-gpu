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
mod title;
pub use title::*;
mod run;
pub use run::*;

#[cfg(test)]
mod tests;
