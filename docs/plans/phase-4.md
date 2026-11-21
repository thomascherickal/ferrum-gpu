# Phase 4: training

- [x] 4.1 Backward kernels, each checked against the finite-difference-verified CPU backward (`ferrum_ref::grad`): dlogits, linear (dX, dW, db), LayerNorm, GELU, attention (dP, dS, dQ, dK, dV), position embedding, token embedding through a CSR index (duplicate-token test, bit-identical to the CPU sum). Tied-head fold is not implemented (see open items).
- [x] 4.2 Fixed-order tree sum of squares, NaN-safe clip scale, AdamW kernel with host-computed lr and bias corrections, decay on matrices and embeddings only.
- [x] 4.3 QAT: snapped copies of the weights feed the forward and backward pass while Adam updates the masters (`GpuTrainer::set_qat`, `ferrum_ref::qat`, `CpuTrainer::with_qat`). The Q8_0 snap kernel is bit-exact against the CPU snap (and against the GGUF encoder) on the MX330, Xe and llvmpipe; a snapped model survives a Q8_0 export exactly. A 25-step QAT run matches the CPU to 6.7e-5. Step overhead 0.1% to 2.3%, see `../train-speed.md`.
- [x] 4.4 A backend-independent training loop (`ferrum_ref::trainer`): strided windows, a per-epoch seeded shuffle that depends only on `(seed, epoch)` so resume needs no saved RNG, warmup + cosine, stop and checkpoint hooks, atomic GGUF checkpoints (`ferrum_model::checkpoint`: weights, both Adam moments, step, loop position, run shape). Both a CPU engine and the GPU trainer run it. Resume from a file on disk is bit-identical to an uninterrupted run, on CPU and on GPU.
- [x] 4.5 Micro-batching: a batch that does not fit the budget is split into the largest fitting divisor, gradients accumulate on the device, one optimizer step follows. Matches the unsplit step to 2.3e-7 after three steps; refused only when one sequence at a time does not fit.
- [x] 4.6 Determinism: two identical runs (including split runs) end byte-identical.
- [x] 4.7 Loss curve: 60 steps through the real loop match the CPU engine to 3.2e-7 per step; the trained models' logits agree to 1.7e-5. (Key-bias weights are excluded from the weight comparison because their true gradient is zero, so Adam amplifies rounding noise.)
- [x] 4.8 Benchmarks: see [`../train-speed.md`](../train-speed.md).

## Result (2026-11-10)

**Gate met.** (Speedups vary run to run on this laptop: the large model measured 5.1x in the main run and 4.2x in a later one; the 2x gate held in both.) A full training step is 5.1x (MX330) and 5.5x (Intel Xe) faster than an 8-thread data-parallel CPU step on a 23.2M-parameter model, and 2.5x / 1.9x ferrum-cpu's own trainer on the reference preset (8,339 tokens/s measured, no QAT). A profile found attention backward at 67% of the first version's step; restructuring it took the large step from 2093 ms to 623 ms.

## Open items

- Weight tying and dropout on the GPU.
- Only a single Vulkan vendor pair was measured; no Apple Silicon, AMD or desktop NVIDIA numbers.
- No full-length convergence run comparing GPU and CPU loss over hundreds of steps on real text; the checked horizon is 60 steps on a synthetic corpus.
