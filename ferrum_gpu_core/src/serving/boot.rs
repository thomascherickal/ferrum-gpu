//! Starting a server from its command line: parse the flags, open the device, load and pre-flight the
//! model, listen. Used by the `ferrum-gpu-server` binary and by the Studio's Serve tab, so both start
//! exactly the same server the same way.
use crate::commands::device::{decide, Workload};
use crate::commands::flags::{switch, val, Args};
use crate::commands::gpt2;
use crate::serving::decoder::{Decoder, Gpt2Decoder, ModelDecoder};
use crate::serving::limits::Limits;
use crate::serving::serve::{check_exposure, start, Config, Running};
use crate::session::{DeviceChoice, Session};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
mod usage;
pub use usage::*;
mod config;
mod decoder;
mod start;
pub use start::*;

#[cfg(test)]
mod tests;
