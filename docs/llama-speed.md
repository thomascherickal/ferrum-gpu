# Phase 5: Llama-family decode, fine-tuning speed and size table (2026-11-11)

**Result: the gate is met, with the same split as GPT-2.** Fine-tuning on the GPU beats an 8-thread CPU step on the two discrete/integrated GPUs once the model is more than tiny (MX330 1.98x to 5.57x, Intel Xe 0.97x to 4.07x), and is *slower* on llvmpipe (software Vulkan, 0.44x to 0.54x), as expected. Single-stream decode is **slower than one CPU thread** on every adapter at every size measured (0.01x to 0.72x); the gap closes as the model grows (large: 0.71x on the MX330). Quantised weights (Q8_0, Q4_0) did **not** speed decode up on the GPUs: the kernel is correct and bit-exact but unoptimised (see below).

## Setup

- Same machine as the earlier reports (8 CPU threads, rustc 1.96.0, `wgpu` 30.0.1, Vulkan, release profile, LTO off). Driver versions not recorded. One run per adapter on 2026-11-11 (04:52 to 04:53 UTC); numbers vary about 20% run to run on this laptop, so read ratios as ranges.
- `cargo bench -p ferrum_wgpu --bench llama` (`FERRUM_GPU_ADAPTER` selects the adapter). **Random weights**: speed and memory depend on the shape, not the values. Shapes (all llama-style, GQA, NORM RoPE, no biases): small = dim 96, 3 layers, 4 heads / 2 KV, vocab 1024 (0.5M params); medium = dim 256, 6 layers, 8 / 2, vocab 4096 (6.3M); large = dim 512, 8 layers, 8 / 4, vocab 8192 (32.0M).
- Decode: 48 cached tokens after a 4-token prompt, median not taken (mean over the 48). CPU = `ferrum_ref::llama::Llama::feed`, one thread (how a CPU runtime runs m=1). GPU = `GpuLlama::feed`, weights, KV cache and activations resident, one command batch per token, one logits row read back.
- Fine-tuning: batch 4 x 32 tokens, f32 AdamW, clip 1.0, mean of 5 steps after one warm-up. CPU baseline = `ferrum_ref::llama::LlamaTrainer` with 8 matmul threads (the tuned matmul; its other ops are plain loops, so it is not ferrum-cpu's code and not a ferrum-cpu measurement). GPU = `GpuLlamaTrainer::step`.

## Decode (ms/token, tokens/s, speed relative to the CPU thread)

| adapter | model | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |
|---|---|---|---|---|---|
| MX330 | small | 0.08 (13325/s) | 1.93 (0.04x) | 2.22 (0.03x) | 1.78 (0.04x) |
| MX330 | medium | 1.65 (607/s) | 4.87 (0.34x) | 6.36 (0.26x) | 5.39 (0.31x) |
| MX330 | large | 7.75 (129/s) | 10.95 (0.71x) | 13.84 (0.56x) | 11.34 (0.68x) |
| Intel Xe | small | 0.07 (13781/s) | 5.95 (0.01x) | 6.58 (0.01x) | 5.18 (0.01x) |
| Intel Xe | medium | 1.52 (660/s) | 9.66 (0.16x) | 10.24 (0.15x) | 9.00 (0.17x) |
| Intel Xe | large | 7.17 (139/s) | 19.30 (0.37x) | 20.03 (0.36x) | 17.00 (0.42x) |
| llvmpipe | small | 0.07 (13855/s) | 3.01 (0.02x) | 2.22 (0.03x) | 2.22 (0.03x) |
| llvmpipe | medium | 1.52 (660/s) | 9.10 (0.17x) | 6.32 (0.24x) | 5.47 (0.28x) |
| llvmpipe | large | 7.54 (133/s) | 23.90 (0.32x) | 13.52 (0.56x) | 10.51 (0.72x) |

Why decode loses: a token is a chain of ~40 to 100 tiny dispatches over a few megabytes of weights, so launch and dependency latency dominate (the same finding as `docs/decode-crossover.md`). The quantised kernel is one invocation per output element with a serial loop over the whole input (no K-split like `gemv.wgsl`), so it moves less memory but exposes less parallelism; on the GPUs that cost more than the saved traffic at these sizes. It helps only on llvmpipe, where memory bandwidth is the limit. A K-split quantised GEMV is the obvious next optimisation (ponytail note in `llama/mod.rs`); the memory saving (3.6x and 6.4x against f32, `ops_quant`) is real regardless of speed.

## Fine-tuning step (ms/step)

| adapter | model | cpu x8 | gpu | gpu vs cpu |
|---|---|---|---|---|
| MX330 | small | 22.9 | 11.6 | 1.98x |
| MX330 | medium | 166.3 | 45.4 | 3.67x |
| MX330 | large | 774.5 | 139.1 | 5.57x |
| Intel Xe | small | 20.4 | 21.0 | 0.97x |
| Intel Xe | medium | 166.4 | 68.3 | 2.44x |
| Intel Xe | large | 751.5 | 184.8 | 4.07x |
| llvmpipe | small | 20.6 | 47.0 | 0.44x |
| llvmpipe | medium | 164.0 | 300.9 | 0.54x |
| llvmpipe | large | 766.3 | 1471.5 | 0.52x |

Training (batched, resident) wins on real GPUs, matching the GPT-2 result (`docs/train-speed.md`). On the smallest shape the Xe only ties the CPU; llvmpipe is a CPU emulation and loses, so it is a correctness adapter, not a speed one.

## Correctness behind the speed numbers

- Forward: GPU logits match the CPU oracle at every position of a full context on MX330, Intel Xe and llvmpipe (4 variants: llama, qwen-style with biases and a tied head, MQA with partial rotary, a width that is not a multiple of 32). Worst relative error observed 4.8e-6 against the 1e-5 forward tolerance (quantised and bias variants are the worst; f32 llama variants are near 1e-6). The margin is about 2x, not 30x.
- Quantised: dequantised weights are bit-identical to `ferrum_gguf::quant::decode` (identity read-back test on all three adapters); products are within 1.3e-6 of the CPU matmul on the dequantised weights.
- Backward: every kernel against the CPU oracle, worst 6.6e-7 (tolerance 1e-5); the CPU oracle's whole-model gradients are checked against finite differences for four variants. Whole-model GPU gradients: worst 7.6e-7.
- Training run: 12 AdamW steps follow the CPU loss curve with a worst per-step loss difference of 3.3e-7 on all three adapters (loss 2.5 to 3.0 falling to 0.02 to 0.06), and the trained models agree on every logit to 2e-5. Parameters whose true gradient is zero (the key biases) are not compared: Adam amplifies rounding noise there, so they legitimately drift.

## Size table (buffer plans, MiB)

Totals are the sums of `llama::plan` (inference: weights, RoPE tables, KV cache for the full context, scratch) and `static_plan_for` + `activation_plan` (fine-tuning: weights, gradients, both Adam moments, activations for batch 1 x 128). The benchmark's "verdict" column only applies the adapter's **per-buffer binding limit** (2047 MiB on MX330 and Xe, 128 MiB on llvmpipe); it does not compare the *total* with the card's memory, so read the totals against your own budget.

| model | params | infer f32 | infer Q8_0 | infer Q4_0 | fine-tune f32 | evidence |
|---|---|---|---|---|---|---|
| small | 0.5M | 2 | 1 | 1 | 13 | run: decode (all four precisions) and fine-tune on all three adapters |
| medium | 6.3M | 25 | 11 | 8 | 119 | run: decode and fine-tune on all three adapters |
| large | 32.0M | 126 | 50 | 37 | 539 | run: decode and fine-tune on all three adapters |
| SmolLM-135M shape | 163M | 712 | 343 | 279 | 2687 | **plan only, never run** |
| Qwen2-0.5B shape | 630M | 2454 | 1099 | 864 | 10047 | **plan only, never run** |
| TinyLlama-1.1B shape | 1100M | 4286 | 1449 | 956 | 17228 | **plan only, never run** |

Only the first three rows are claimed. The plan-only rows are arithmetic: they say, for example, that on a 2 GB MX330 the 0.5B and 1.1B shapes would not fit in f32 and that quantisation is what would bring inference near the card's size. Nothing larger than 32M parameters was executed, and no third-party checkpoint was loaded. The runs above used random weights, so they say nothing about output quality.

## What was not run

Apple Silicon, AMD and desktop NVIDIA; any model above 32M parameters; micro-batched, tied-head or QAT fine-tuning of Llama (not implemented on the Llama trainer); a real GGUF produced by ferrum-cpu or llama.cpp (Phase 6 fixture); driver versions were not recorded.
