// dP[b,h,i,j] = dO[i] . V[j] for j <= i (0 beyond the diagonal), one invocation per cell.
// Written into the `ds` scratch, which later kernels turn into dS in place.
struct U { batch: u32, t: u32, heads: u32, hd: u32 }
//! use common/attn_dot A=dout B=v OUT=ds KV=u.heads SCALE=1.0
