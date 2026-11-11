//! Request limits and the fixed server-side caps. Defaults and names follow ferrum-cpu's
//! `ferrum-serve` (docs/PARITY.md).

/// Most POST requests the server holds at once; more get an immediate 503.
// ponytail: fixed; make it a `Limits` field if a multi-client deployment needs more.
pub const MAX_INFLIGHT: usize = 32;
/// Most stop strings per generate request, and the longest one (bytes).
pub const MAX_STOPS: usize = 16;
/// Longest stop string (bytes).
pub const MAX_STOP_BYTES: usize = 1024;
/// `max_tokens` when a request omits it (capped by the server limit).
pub const DEFAULT_MAX_TOKENS: usize = 64;

/// Limits checked before any model compute.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Most requests in one batched call.
    pub max_batch: usize,
    /// Most characters in one prompt, context or text.
    pub max_chars: usize,
    /// Largest request body (bytes).
    pub max_body: usize,
    /// Cap on `max_tokens` per generate request.
    pub max_tokens: usize,
    /// Repetition penalty for a generate request that omits the field.
    pub repetition_penalty: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_batch: 64,
            max_chars: 65_536,
            max_body: 8 << 20,
            max_tokens: 4096,
            repetition_penalty: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_ferrum_cpu() {
        let d = Limits::default();
        assert_eq!(
            (d.max_batch, d.max_chars, d.max_body, d.max_tokens),
            (64, 65_536, 8 << 20, 4096)
        );
        assert_eq!(d.repetition_penalty, 1.0);
        assert_eq!((MAX_INFLIGHT, MAX_STOPS, MAX_STOP_BYTES), (32, 16, 1024));
    }
}
