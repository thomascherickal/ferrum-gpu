//! Training on the device: GPT-2 and Llama/Qwen2 trainers.

mod batch_check;
mod device_params;
pub mod gpt2;
pub mod llama;
pub(crate) use device_params::DeviceParams;
mod dims;
pub(crate) use dims::Dims;
mod head_acts;
pub(crate) use head_acts::HeadActs;
