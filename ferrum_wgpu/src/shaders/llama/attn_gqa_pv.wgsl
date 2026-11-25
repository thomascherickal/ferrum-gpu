// out[(b*t+i), h*hd+d] = sum_{j<=i} probs[b,h,i,j] * v[(b*t+j), kvh*hd+d]; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_rows P=probs V=v OUT=out KV=u.kv_heads
