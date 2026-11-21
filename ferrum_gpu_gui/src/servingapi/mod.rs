//! The Serve and Export tabs' API: run the JSON server in this process and stop it, and say which
//! export tools (`ollama`, `lms`) are installed. The server and its startup are in
//! `ferrum_gpu_core::serving`; the tool lookup is the one the export commands use.
//!
//! | route | does |
//! |---|---|
//! | `GET /api/serve/status` | whether a server runs, its address, whether it needs a token, and its status lines |
//! | `POST /api/serve/start` `{"args":["--model","m.gguf","--port","8080"]}` | start it with the `ferrum-gpu-server` flags |
//! | `POST /api/serve/stop` | stop it |
//! | `GET /api/tools` | `ollama` and `lms`: where each is found, or why not |
use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;
use ferrum_gpu_core::commands::runtimes::find_tool;
use ferrum_gpu_core::serving::host::Status;
mod handles;
pub use handles::*;
mod route;
pub use route::*;

#[cfg(test)]
mod tests;
