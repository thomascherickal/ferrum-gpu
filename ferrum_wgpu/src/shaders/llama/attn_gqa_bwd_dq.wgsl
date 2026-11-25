// dQ[(b,i), h*hd+d] = sum_{j<=i} dS[b,h,i,j] * K[(b,j), kvh*hd+d], ascending j; one invocation
// per output element.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_rows P=ds V=k OUT=dq KV=u.kv_heads
