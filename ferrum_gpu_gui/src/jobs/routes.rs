//! The HTTP routes of the jobs API:
//!
//! | route | does |
//! |---|---|
//! | `GET /api/jobs` | every remembered job, one summary each |
//! | `POST /api/jobs` `{"cmd":"train","args":["--data","t.txt",...]}` | start one; `{"id":N}` |
//! | `GET /api/jobs/N?since=K` | state, exit status, output lines from index K, last progress |
//! | `POST /api/jobs/N/stop` | ask it to stop; training and generation end cleanly at the next step or token |
use super::view;
use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;
mod handles;
pub use handles::*;
mod route;
pub use route::*;
