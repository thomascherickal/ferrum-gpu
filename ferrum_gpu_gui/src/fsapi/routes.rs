//! The HTTP routes of the file API (all `GET`):
//!
//! | route | answers |
//! |---|---|
//! | `/api/fs/list?dir=P` | the entries of folder P (default: where the server was started), and the places to start from |
//! | `/api/fs/stat?path=P` | `file`, `dir` or `missing`, with file count and bytes |
//! | `/api/fs/preview?path=P&bytes=N` | the first N bytes of a file as text, its size and line count |
use super::{list, preview, stat, MAX_PREVIEW};
use crate::http::{Request, Response};
use crate::json::Json;
use std::path::{Path, PathBuf};
mod handles;
pub use handles::*;
mod route;
pub use route::*;
