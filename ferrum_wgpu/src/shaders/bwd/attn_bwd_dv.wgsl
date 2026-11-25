// dV[(b,j), col] = sum_{i>=j} P[b,h,i,j] * dO[(b,i), col], ascending i; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_cols P=probs X=dout OUT=dv KV=u.heads
