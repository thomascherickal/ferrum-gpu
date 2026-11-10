# Phase 3: single-token decode, CPU vs GPU (2026-11-10)

**Result: on these GPUs, cached single-token decode is not faster than one CPU thread at any size tested (0.5M to 33.7M parameters).** The GPU decoder is correct (it matches the CPU oracle at every position of a full context to 1e-6), but speed is another matter. The `auto` device rule must keep single-stream decode on the CPU for models in this range.

## Setup

- Same machine as the Phase 0 report (8 CPU threads, rustc 1.96.0, `wgpu` 30.0.1, Vulkan, release profile). Driver versions not recorded.
- `cargo bench -p ferrum_wgpu --bench decode`, std-only. Random weights; 4 warm-up tokens, then 48 decoded tokens timed as a whole; one run per adapter, so small differences are noise.
- CPU: `ferrum_ref::gpt2::Gpt2::feed`, one thread, f32, cached. That is how ferrum-cpu runs an `m = 1` f32 decode (its own `gemm` bench labels it "SERIAL on one core, bandwidth-bound"). Multi-threaded and quantised CPU decode were **not** measured.
- GPU: `GpuGpt2::feed`: weights, KV cache and activations resident, the whole token recorded into one command encoder, one index in and one logits row out per token.
- Sizes: small = the 507,552-parameter reference preset; medium = dim 256, 6 layers; large = dim 512, 8 layers, vocab 8192 (33.7M parameters including the embedding).

## Results

Final kernels (16-way K-split GEMV):

### NVIDIA GeForce MX330 (Vulkan, discrete)

| model | params | cpu ms/token | cpu tok/s | gpu ms/token | gpu tok/s | gpu vs cpu |
|---|---|---|---|---|---|---|
| small (reference preset) | 0.5M | 0.084 | 11862 | 1.685 | 593 | 0.05x |
| medium | 6.9M | 1.792 | 558 | 4.587 | 218 | 0.39x |
| large | 33.7M | 10.060 | 99 | 10.872 | 92 | 0.93x |

### Intel Iris Xe Graphics (Vulkan, integrated)

| model | params | cpu ms/token | cpu tok/s | gpu ms/token | gpu tok/s | gpu vs cpu |
|---|---|---|---|---|---|---|
| small (reference preset) | 0.5M | 0.083 | 12054 | 4.996 | 200 | 0.02x |
| medium | 6.9M | 1.839 | 544 | 9.120 | 110 | 0.20x |
| large | 33.7M | 9.240 | 108 | 20.549 | 49 | 0.45x |

## What the numbers say

1. **A latency floor, not a compute limit, decides small models.** A temporary probe on the MX330 (release build, reference preset) split one token into 0.2 ms of host recording, 0.3 ms of submit and 1.6 ms of waiting for the GPU. That wait covers about 50 dependent tiny commands, roughly 25 microseconds each. The CPU does the whole token in 0.084 ms. Fusing kernels would shave the floor but cannot remove it.
2. **Batching into one submit did not help** (2.47 ms to 2.62 ms for the reference preset, within noise): submit count was not the bottleneck. The batch API stays because it is correct and removes per-kernel error-scope work, not because it is faster here.
3. **The GEMV kernel mattered for large models.** The first version (one invocation per output column) left the GPU latency-bound: 19.7 ms per token for the large model, about 7 GB/s. Splitting K across 8 and then 16 slices per workgroup, with a fixed-order combine, brought it to 10.9 ms (0.50x to 0.93x of CPU). It is still below parity and the Xe benefits less.
4. **Where the GPU should win is batched work, not this.** The Phase 0 and Phase 2 matmul measurements show 3x to 7x over the tuned 8-thread CPU kernel at 1024 and larger, but only for resident data. Whether whole-model prefill, scoring and training steps cross over was **not** measured at model level and is Phase 4's question.

## Consequences for the plan

- `auto` keeps interactive single-stream decode on the CPU for models up to at least 34M parameters on these two GPUs.
- Do not claim a decode speedup for Apple Silicon, AMD or discrete desktop parts; none were run, and a faster GPU lowers the per-dispatch floor only somewhat.
- Open levers, in order of expected value: fewer commands per token (fused LayerNorm+GEMV, GEMV+GELU), vec4 weight loads, quantised in-kernel weights (4x less memory traffic for the bandwidth-bound large case), and multi-token decode (speculative or batched requests), which turns GEMV into matmul.
