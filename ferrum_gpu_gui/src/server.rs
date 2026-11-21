//! Starting the server and parsing its command line.
use crate::api::{route, MAX_BODY};
use crate::app::{App, DeviceChoice};
use crate::http::{serve, Handler};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
mod options;
pub use options::*;
mod start;
pub use start::*;

#[cfg(test)]
mod tests;
