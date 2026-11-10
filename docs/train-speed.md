# Phase 4: training step speed, CPU vs GPU (2026-11-10)

**Result: the gate is met.** A full GPT-2 training step (forward, backward, global-norm clip, AdamW) is 5.1x (MX330) and 5.5x (Intel Xe) faster than an 8-thread data-parallel CPU step on the 23.2M-parameter model, and 1.3x to 1.7x faster even on the tiny reference preset. Against ferrum-cpu's real trainer on the reference preset the GPU is 2.5x (MX330) and 1.9x (Xe). The GPU trainer is bit-identical across repeat runs.

This is the opposite of the Phase 3 decode result, and consistent with it: batched work with resident data wins, one-token-at-a-time decode does not.

## Setup

- Same machine as the earlier reports (8 CPU threads, rustc 1.96.0, `wgpu` 30.0.1, Vulkan, release). Driver versions not recorded. Median of 3 to 10 steps after one warm-up; one session per adapter.
- `cargo bench -p ferrum_wgpu --bench train`. Random-token batches (the speed of a step does not depend on token values). Gradient clip 1.0, AdamW with weight decay 0.01.
- **GPU:** `GpuTrainer::step`: weights, gradients, Adam moments and activations resident; one command batch per step; only the loss and gradient norm are read back.
- **CPU (data-parallel):** `ferrum_ref`, written in this repo: the batch is split into 8 shards, each thread runs forward and backward on its shard with the tuned matmul, the gradients are averaged, then AdamW. This mirrors ferrum-cpu's own design (it also shards the batch across threads), so every op scales with the thread count. Its non-matmul ops are simple loops, so it is **not** ferrum-cpu's code.
- **ferrum-cpu (real):** the shipped `train_transformer train` binary, built outside that repo, run read-only: reference preset (context 128, embed 96, 4 heads, 3 blocks, hidden 320, vocab 1024), `--tokens 4e5 --lr 5e-3 --cosine --warmup 30 --clip 1.0 --no-qat --threads 8`, on a synthetic 320 KB text corpus. 8,339 tokens/s end to end (47.95 s of training, wall clock 48.0 s; includes its tokenizer training and data loading). QAT was off because the GPU trainer has none yet.
- Sizes: small = reference preset (0.5M params, batch 16 x 128); medium = dim 256, 4 layers (5.3M, batch 8 x 128); large = dim 512, 6 layers, hidden 2048, vocab 4096 (23.2M, batch 8 x 128).

## Results

### NVIDIA GeForce MX330

| model | params | batch x t | cpu x1 ms/step | cpu x8 data-parallel ms/step | gpu ms/step | gpu tok/s | gpu vs cpu x8 |
|---|---|---|---|---|---|---|---|
| small (reference preset) | 0.5M | 16 x 128 | 595.9 | 173.2 | 99.9 | 20496 | 1.73x |
| medium | 5.3M | 8 x 128 | 2081.4 | 708.8 | 175.5 | 5834 | 4.04x |
| large | 23.2M | 8 x 128 | 9237.6 | 3173.5 | 623.2 | 1643 | 5.09x |

### Intel Iris Xe Graphics

| model | params | batch x t | cpu x1 ms/step | cpu x8 data-parallel ms/step | gpu ms/step | gpu tok/s | gpu vs cpu x8 |
|---|---|---|---|---|---|---|---|
| small (reference preset) | 0.5M | 16 x 128 | 604.3 | 172.8 | 130.5 | 15696 | 1.32x |
| medium | 5.3M | 8 x 128 | 1983.2 | 699.9 | 229.3 | 4467 | 3.05x |
| large | 23.2M | 8 x 128 | 9093.5 | 3383.5 | 619.1 | 1654 | 5.47x |

### Against ferrum-cpu's real trainer (reference preset, tokens per second)

| trainer | tokens/s | vs ferrum-cpu |
|---|---|---|
| ferrum-cpu `train`, 8 threads, no QAT | 8339 | 1.00x |
| `ferrum_ref` data-parallel (the baseline above) | 11830 | 1.42x |
| GPU, MX330 | 20496 | 2.46x |
| GPU, Intel Xe | 15696 | 1.88x |

The `ferrum_ref` baseline is faster than the real ferrum-cpu trainer on this shape, so the GPU speedups in the tables above are conservative with respect to ferrum-cpu. The comparison is not exact: ferrum-cpu's figure includes tokenizer training and I/O, its default batch size and step count were not matched, and its run used real BPE tokens on a synthetic corpus.

## How the step got fast (a profiling story worth keeping)

The first GPU trainer was correct but slow: **2093 ms per large step, only 1.49x the CPU, short of the 2x gate**. A per-kernel profile (`benches/profile_train.rs`, each kernel alone at the large shapes) summed to 2127 ms, matching the measured step, and showed where the time went:

| kernel | before (ms each) | after (ms each) |
|---|---|---|
| attention backward | 238.8 (67% of the step) | 13.4 |
| attention forward | 25.3 | 9.0 |
| all matmuls together (per step) | about 450 | about 450 |
| estimated step | 2127 | 665 |

Attention backward recomputed the full `dO . V` dot product for every output element, a 64x redundancy. It now computes the score-gradient matrix once, then dQ, dK and dV are cheap gathers. Attention forward computes all scores in parallel (one invocation per cell) and softmaxes per row. Both changes keep every sum in the same ascending order, so every parity test still passes at the same tolerance. After the fix the matmuls are about 70% of a step, so the remaining speed comes from the matmul kernel.

## What this does not show

- **Quantisation-aware training, dropout and weight tying are not implemented on the GPU**, while ferrum-cpu has all three; the GPU numbers are for the plain AdamW path. QAT would add a snap pass per step.
- Larger batches or longer contexts, other GPUs (Apple Silicon, AMD, NVIDIA desktop parts) and multi-GPU were not run.
- Convergence (loss curve against the CPU trainer over hundreds of steps) is covered by the unit tests only at the level of a few steps matching to 1e-6 and a memorisation test; a full loss-curve comparison is Phase 4 task 4.7.
- Memory: the large model's weights, gradients and both Adam moments alone are 23.2M x 4 buffers x 4 bytes, about 371 MB. Activation memory was not measured (the trainer plans it, and the budget refusal path is tested), and no run near the MX330's 2 GB was made.
