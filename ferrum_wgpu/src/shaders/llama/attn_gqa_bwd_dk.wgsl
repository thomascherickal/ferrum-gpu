// dK[(b,j), kvh*hd+d] = sum over the query heads h of the group, then i >= j (ascending), of
// dS[b,h,i,j] * Q[(b,i), h*hd+d]. One invocation per output element.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_cols P=ds X=q OUT=dk KV=u.kv_heads
