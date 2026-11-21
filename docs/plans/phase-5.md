# Phase 5: imported Llama/Qwen models

- [x] 5.0 CPU oracle (`ferrum_ref::llama`: forward, KV-cached decode, backward, AdamW) and the `llama`/`qwen2` GGUF mapping (`ferrum_model::llama_gguf`), with a synthetic fixture built by our own writer.
- [x] 5.1 Kernels with parity: RMSNorm, RoPE (host-built cos/sin tables; NORM and NEOX), GQA, SwiGLU.
- [x] 5.2 Q8_0/Q4_0 linear: weights packed in `u32`, dequantised in the inner loop, vs reference dequant.
- [x] 5.3 `GpuLlama` decoder (device KV cache, `forward_all`, `score`, budget refusal, fault sweeps).
- [x] 5.4 f32 AdamW fine-tuning: backward kernels vs the CPU oracle (which is itself checked against finite differences), `GpuLlamaTrainer`.
- [x] 5.5 Size table from the buffer plan, claimed only where a real run exists (`docs/llama-speed.md`).

**Gate:** tiny-llama parity on every adapter in the matrix (met: MX330, Intel Xe, llvmpipe; no adapter skips cleanly), GPU fine-tune loss curve matches the CPU oracle (met: 12 steps, worst loss difference about 3e-7), measured speed comparison in `docs/llama-speed.md`.

Open (carried into later work): the shared `Engine` trait, the training loop and the checkpoint format are still GPT-2 shaped, so Llama fine-tuning has its own `step` and `LlamaSnapshot` instead of plugging into them; tied heads, micro-batching, QAT and dropout are not on the Llama trainer; no real third-party checkpoint has been loaded (cross-loading of a ferrum-cpu-produced GGUF is a Phase 6 fixture).
