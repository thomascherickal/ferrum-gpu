//! A minimal HTTP/1.1 server (std only): one request per connection, `Content-Length` bodies,
//! hard limits on header and body size, read timeouts, and a Host/Origin check against DNS
//! rebinding and cross-site requests (the server only ever listens on loopback).
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
mod max_head;
pub use max_head::*;
mod read_request_m;
pub use read_request_m::*;
mod serve_with_m;
pub use serve_with_m::*;

#[cfg(test)]
mod tests;
