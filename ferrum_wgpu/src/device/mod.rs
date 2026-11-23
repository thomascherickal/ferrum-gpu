//! Real adapter enumeration and device opening. All direct wgpu device calls live in this file,
//! `tensor.rs` and `matmul.rs` so a wgpu upgrade touches three places.

use crate::select::{choose, AdapterSummary, Kind, SelectError};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
pub mod budget;
mod gpuerror;
pub mod memory;
pub mod runtime;
pub mod select;
pub mod tensor;
#[cfg(any(test, feature = "fault-injection"))]
pub(crate) use runtime::Faults;
mod error_from_text;
pub(crate) use error_from_text::err;
mod stage;
pub use gpuerror::*;
pub(crate) use stage::stage;
mod wait_for_loss;
pub use wait_for_loss::*;

#[cfg(test)]
mod tests;
