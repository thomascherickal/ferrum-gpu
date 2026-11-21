//! `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`.
//! Request and response shapes follow ferrum-cpu's `ferrum-serve` so clients written for it work
//! unchanged:
//!
//! - `POST /v1/loglikelihood` `{"requests":[{"context","continuation"}]}` gives
//!   `{"results":[{"logprob","n_tokens","greedy"}]}`;
//! - `POST /v1/loglikelihood_rolling` `{"texts":[...]}` gives `{"results":[{"logprob","n_tokens"}]}`;
//! - `POST /v1/generate` `{"requests":[{"prompt","max_tokens","stop","temperature","top_k","top_p",
//!   "seed","repetition_penalty"}]}` gives `{"results":[{"text","n_tokens"}]}`.
use crate::http::{Request, Response};
use crate::json::Json;
use crate::serving::decoder::{Decoder, GenReq};
use crate::serving::limits::{Limits, DEFAULT_MAX_TOKENS, MAX_STOPS, MAX_STOP_BYTES};
use crate::serving::score::{pair, rolling};
mod num;
pub(crate) use num::*;
mod route;
pub use route::*;
mod loglikelihood;
pub(crate) use loglikelihood::*;
mod loglikelihood_rolling;
pub(crate) use loglikelihood_rolling::*;
mod gen_request;
pub(crate) use gen_request::*;
mod generate;
pub(crate) use generate::*;

#[cfg(test)]
mod tests;
