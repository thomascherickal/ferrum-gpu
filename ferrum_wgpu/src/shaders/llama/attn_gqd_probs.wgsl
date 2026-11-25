// Single-query grouped-query attention probabilities over a KV cache laid out [ctx, kv_heads*hd].
// One invocation per query head; positions 0..len are visible; probs is [heads, ctx].
struct U { heads: u32, kv_heads: u32, hd: u32, len: u32, ctx: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_dec_probs KV=u.kv_heads
