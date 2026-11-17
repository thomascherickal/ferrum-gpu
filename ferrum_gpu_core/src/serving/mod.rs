//! The JSON server (`ferrum-gpu-server`) as a library: routes, limits, bearer-token auth, decoders over
//! a loaded model, and [`boot`], which turns the server's command line into a running server. The
//! `ferrum-gpu-server` binary is a thin `main` over [`boot::boot`], and the Studio runs the same server
//! in-process.

pub mod auth;
pub mod boot;
pub mod decoder;
pub mod host;
pub mod limits;
pub mod routes;
pub mod score;
pub mod serve;
