#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! The `ferrum-gpu-server` (Phase 6.2): a small JSON API over a [`decoder::Decoder`].

pub mod auth;
pub mod decoder;
pub mod limits;
pub mod routes;
pub mod score;
pub mod serve;
