#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! The `ferrum-gpu-server` (Phase 6.2): a small JSON API over a [`decoder::Decoder`].

pub mod decoder;
pub mod routes;
pub mod serve;
