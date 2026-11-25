// dQ[(b,i), col] = sum_{j<=i} dS[b,h,i,j] * K[(b,j), col], ascending j; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_rows P=ds V=k OUT=dq KV=u.heads
