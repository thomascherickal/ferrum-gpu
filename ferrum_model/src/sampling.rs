//! Token-level sampling from a logits row (the same semantics ferrum-cpu documents:
//! temperature 0 is exact argmax with the first maximum winning).
/// The shared seeded generator (lives in `ferrum_ref`, re-exported for callers of this module).
pub use ferrum_ref::rng::Rng;
mod params;
pub use params::*;
mod sample;
pub use sample::*;

#[cfg(test)]
mod tests;
