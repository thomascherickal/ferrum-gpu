# ferrum-gpu: initial plan

A GPU-capable sibling of [ferrum-cpu](../ferrum-cpu/ReadMe.md) (the CPU engine, repo `Ferrum`). Status: **plan only, no code yet.**

## Goal
Train, fine-tune and run the same small Transformers on a GPU, with the same GGUF files, CLI, server and GUI behaviour. A model trained on either side loads on the other. The CPU path stays the reference.

## Principles (inherited from ferrum-cpu)
- GGUF only; no new file formats. Models stay runnable in llama.cpp, Ollama and LM Studio.
- `#![forbid(unsafe_code)]` in our own crates. Never call a dependency's `unsafe fn` directly.
- Every new dependency needs a measured gain, a determinism test, `cargo deny`, and a `wasm32` build of the shared core.
- Tests first, tiny models, commit per feature, push after each green suite.
- CPU fallback always works; no GPU, no problem.

## Open decisions (resolve in Phase 0)
| Question | Leading option | Alternative |
|---|---|---|
| Backend API | `wgpu` + WGSL compute shaders: portable (Vulkan/Metal/DX12), safe Rust surface, also reaches the browser (WebGPU) | CUDA through `cudarc` (fastest on NVIDIA, FFI heavy, NVIDIA only); Vulkan via `ash` (unsafe-heavy) |
| Repo shape | New crate(s) in this directory depending on `ferrum_core` by path (`../ferrum-cpu/ferrum_core`), like `Ferrum-ML` | Feature-gated backend inside `ferrum-cpu` |
| Determinism | Fixed reduction order per kernel, so repeat runs on the same GPU/driver are bit-identical; cross-device equality is a non-goal | Tolerance-only parity with CPU |
| Precision | f32 first; f16/bf16 and int8 weights later | |

## Hardware reality check
Dev box: NVIDIA GeForce MX330 with 2 GB VRAM and an Intel iGPU. Plan for small models first: tiny and small presets (up to roughly 20M parameters) fit in 2 GB. Larger sizes need a bigger card; the memory estimator must refuse honestly, as the CPU one does.

## Phases
0. **Spike (1-2 days):** a `wgpu` matmul kernel (tiled, shared memory) benchmarked against ferrum-cpu's 4x8 CPU kernel on this machine, at 256, 1024 and 2048 squares, including upload/download cost. Decide the backend. Go/no-go gate: at least 3x the 8-thread CPU kernel at 1024+, otherwise stop and write down why.
1. **Backend trait:** a small `Backend` abstraction inside the new crate (matmul, add, GELU, LayerNorm, softmax, embedding, cross-entropy). CPU implementation wraps `ferrum_core::ops` so parity tests have an oracle.
2. **Inference:** forward pass and KV-cached decode for GPT-2-style and Llama-style GGUF models on GPU; `generate`, `eval`, `run-gguf` equivalents; parity with CPU within a stated tolerance on tiny models.
3. **Training:** backward kernels (matmul grads, LayerNorm, GELU, attention, softmax-CE), Adam on device, gradient clipping, QAT snap; training a tiny model to the same loss curve as CPU (tolerance), deterministic across repeat runs.
4. **Memory and scheduling:** VRAM estimator and pre-flight (reuse the GUI/CLI estimate pattern), micro-batching and activation recomputation when it does not fit, streaming token windows from a `ferrum-tokens` file.
5. **Integration:** `--device cpu|gpu|auto` flag in the CLI and server, device selector in the GUI (hardware panel already reports GPUs as "detected, not used"), CI job on a software Vulkan device (lavapipe) for correctness; real-GPU benchmarks run locally.
6. **Quantised inference:** int8/int4 weight kernels (dequantise in-kernel), then optional f16.

## Testing plan
- Unit tests per kernel against the CPU oracle on random and adversarial shapes (odd sizes, 1-row decode, large K).
- Tiny end-to-end tests (train 2 epochs, generate, export, reload) on GPU and on lavapipe in CI.
- Determinism test: two runs, same bytes. Regression test: GPU and CPU final loss within tolerance.
- Benchmarks recorded with machine, driver and date; no number is quoted unless measured.

## Risks
- Driver and shader-compiler variance makes bit-exact claims fragile: document exactly what is guaranteed.
- PCIe transfer can eat the gain for small models: keep tensors resident, upload once per step.
- 2 GB VRAM limits what can be tested locally; CI cannot run a real GPU.
- WebGPU in the browser is a later bonus, not a Phase 1-5 goal.

## First actions
1. Create `ferrum-gpu` as a git repo (`master`), cargo workspace, and a `bench_matmul` spike crate.
2. Write the spike report as `docs/spike-matmul.md` with the go/no-go decision.
3. Only then start Phase 1.
