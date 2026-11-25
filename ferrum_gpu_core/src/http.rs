//! A minimal HTTP/1.1 server (std only): one request per connection, `Content-Length` bodies,
//! hard limits on header and body size, a total time budget per request, a cap on connections, and a Host/Origin check against DNS
//! rebinding and cross-site requests (the server only ever listens on loopback).
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
mod budget;
pub use budget::*;
mod message;
pub use message::*;
mod bad;
pub use bad::*;
mod head;
mod read_request;
pub use read_request::*;
mod write_response;
pub use write_response::*;
mod close;
use close::*;
mod serve_with;
pub use serve_with::*;
mod handle_watched;
pub(crate) use handle_watched::*;

#[cfg(test)]
mod tests;
