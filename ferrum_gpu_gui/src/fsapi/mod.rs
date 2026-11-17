//! The file browser API over HTTP. Listing, summarising and previewing are in
//! `ferrum_gpu_core::fsapi`, re-exported here; this module adds only the routes.

pub use ferrum_gpu_core::fsapi::*;
pub mod routes;

#[cfg(test)]
mod tests;
