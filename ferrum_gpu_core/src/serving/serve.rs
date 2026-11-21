//! Start the server around any [`Decoder`]: bind, optional bearer token, in-flight cap.
use crate::http::{serve_with, Handler, Response};
use crate::serving::auth;
use crate::serving::decoder::Decoder;
use crate::serving::limits::{Limits, MAX_INFLIGHT};
use crate::serving::routes::route;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
mod config;
pub use config::*;
mod start;
pub use start::*;

#[cfg(test)]
mod tests;
