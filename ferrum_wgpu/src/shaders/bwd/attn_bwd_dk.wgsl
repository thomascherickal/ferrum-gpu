// dK[(b,j), col] = sum_{i>=j} dS[b,h,i,j] * Q[(b,i), col], ascending i; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_cols P=ds X=q OUT=dk KV=u.heads
