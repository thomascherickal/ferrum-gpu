// Single-query attention probabilities over a KV cache. One invocation per head; positions
// 0..len are visible. probs is [heads, ctx]; only the first len entries of each row are written.
struct U { heads: u32, hd: u32, len: u32, ctx: u32 }
//! use common/attn_dec_probs KV=u.heads
