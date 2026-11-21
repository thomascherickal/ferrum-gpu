use super::*;

#[test]
fn ollama_on_the_gpu_through_cuda_gives_our_text() {
    run(Mode::Cuda, "gpu-cuda");
}

#[test]
fn ollama_on_the_gpu_through_vulkan_gives_our_text() {
    run(Mode::Vulkan, "gpu-vulkan");
}
