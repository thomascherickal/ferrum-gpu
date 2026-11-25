// Grouped-query causal scores, one invocation per (batch, head, query, key) cell. Query head h
// reads KV head h / (heads / kv_heads). Cells with key > query get 0 (and stay 0 after softmax).
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_dot A=q B=k OUT=probs KV=u.kv_heads SCALE=(1.0/sqrt(f32(u.hd)))
