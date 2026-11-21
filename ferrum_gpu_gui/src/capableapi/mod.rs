//! The Capable tab's API: what a given amount of memory can hold, and what this machine can hold.
//! The formulas are `ferrum_gpu_core::capacity` (the ones in `hardware-caps.md`); everything is an estimate.
//!
//! | route | answers |
//! |---|---|
//! | `GET /api/capable?memory_gb=16&unified=0&bandwidth=448&model_b=7` | the largest model per precision for that memory, and the decode-speed ceiling for a model of `model_b` billion weights at `bandwidth` GB/s (both optional) |
//! | `GET /api/system` | this machine: OS, CPU threads, RAM, and for each adapter its memory budget and what that budget can hold |

use crate::app::App;
use crate::http::{Request, Response};
use crate::json::Json;
use ferrum_gpu_core::capacity::{
    decode_ceiling_tps, format_params, max_params_inference, max_params_training, row, row_json,
    Precision,
};
use ferrum_gpu_core::commands::sysmem::{budget_for, facts_for, System};
mod handles;
mod meter;
pub use handles::*;
mod system;
use system::*;
mod route;
pub use route::*;

#[cfg(test)]
mod tests;
