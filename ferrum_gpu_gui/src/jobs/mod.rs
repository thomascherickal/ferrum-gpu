//! The jobs API over HTTP. The jobs themselves (threads running the commands in-process, with their
//! output, progress and cancellation) are in `ferrum_gpu_core::jobs`, re-exported here; this module
//! adds only the routes.

pub use ferrum_gpu_core::jobs::*;
pub mod routes;

#[cfg(test)]
mod tests;
