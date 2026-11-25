// Scaled attention scores, one invocation per (batch, head, query, key) cell so the whole
// t x t matrix is computed in parallel. Cells with key > query (the causal mask) get 0 here and
// stay exactly 0 after the softmax. Each dot product sums d in ascending order.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_dot A=q B=k OUT=probs KV=u.heads SCALE=(1.0/sqrt(f32(u.hd)))
