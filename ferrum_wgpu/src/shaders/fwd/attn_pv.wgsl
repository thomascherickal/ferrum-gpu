// out[(b*t+i), h*hd+d] = sum_{j<=i} probs[b,h,i,j] * v[(b*t+j), h*hd+d]; one invocation per output.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_rows P=probs V=v OUT=out KV=u.heads
