// dP[b,h,i,j] = dO[i, head h] . V[j, kv head of h] for j <= i (0 beyond the diagonal), one
// invocation per cell. Written into the `ds` scratch, which later kernels turn into dS in place.
struct U { batch: u32, t: u32, heads: u32, hd: u32, kv_heads: u32, p0: u32, p1: u32, p2: u32 }
//! use common/attn_dot A=dout B=v OUT=ds KV=u.kv_heads SCALE=1.0
