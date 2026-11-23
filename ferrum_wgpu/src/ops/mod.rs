//! Forward kernels of the GPT-2 stack. Every function records one or more dispatches on
//! resident buffers and does not wait; read results with [`Gpu::read`].
//! ponytail: 1-D kernels cap at 65,535 workgroups x 64 invocations (4.1M elements) per call;
//! add a 2-D split when a tensor needs more.

/// `pipe!(gpu, "group/name")`: the cached compute pipeline of `shaders/group/name.wgsl`. A macro
/// (not a function) because `include_str!` needs a literal path; it keeps every call site on one
/// short line.
macro_rules! pipe {
    ($gpu:expr, $name:literal) => {
        $gpu.pipeline(
            $name,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/shaders/",
                $name,
                ".wgsl"
            )),
        )
    };
}

mod attention;
pub mod bwd;
pub mod dropout;
mod elementwise;
mod embed;
pub mod llama;
pub mod llama_bwd;
mod loss;
pub mod matmul;
pub mod opt;
pub mod qat;
pub mod quant;

pub use attention::{attention, attention_decode};
pub use elementwise::{add, add_bias, gelu, layernorm};
pub use embed::{embed, embed_at, gemv};
pub use loss::{cross_entropy, score};

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

pub(crate) use crate::device::err;

pub(crate) fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(err(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

impl Gpu {
    /// Upload integer indices that must all be `< bound` (a GPU gather would silently clamp).
    pub fn upload_indices(&self, ids: &[usize], bound: usize) -> Result<GpuBuf, GpuError> {
        if let Some(bad) = ids.iter().find(|&&i| i >= bound) {
            return Err(err(format!(
                "index {bad} is out of range for {bound} entries"
            )));
        }
        let words: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.upload_raw(bytemuck::cast_slice(&words), words.len())
    }
}

#[cfg(test)]
mod tests;
