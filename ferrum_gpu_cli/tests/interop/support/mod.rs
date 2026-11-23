//! Shared helpers: finding the tools, running them under a deadline, and the small trained models
//! every test uses.

pub(crate) mod llama;
pub(crate) mod log;
pub(crate) mod model;
pub(crate) mod ollama_server;
pub(crate) mod proc;
pub(crate) mod tools;
pub(crate) mod wide;

/// Print why a test is skipped and return from it: `skip!("no llama.cpp")`.
#[macro_export]
macro_rules! skip {
    ($($why:tt)*) => {{
        eprintln!("SKIP: {}", format!($($why)*));
        return;
    }};
}
