//! Shared helpers: finding the tools, running them under a deadline, and the small trained models
//! every test uses.

pub mod log;
pub mod model;
pub mod ollama_server;
pub mod proc;
pub mod tools;

/// Print why a test is skipped and return from it: `skip!("no llama.cpp")`.
#[macro_export]
macro_rules! skip {
    ($($why:tt)*) => {{
        eprintln!("SKIP: {}", format!($($why)*));
        return;
    }};
}
