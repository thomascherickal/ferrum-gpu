//! Static files: the page, its script and style, the browser runtime modules (Phase 5b) and the
//! WGSL kernels. Everything is compiled in; lookups are by exact name, so no request path ever
//! reaches the file system.
use crate::http::Response;
mod csp;
pub use csp::*;
mod serve;
pub use serve::*;
