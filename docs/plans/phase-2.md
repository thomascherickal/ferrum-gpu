# Phase 2: forward kernels + CPU reference

Copy into `ferrum_ref` as needed (provenance headers): LayerNorm (eps 1e-5), tanh-GELU, causal attention (`-1e9` mask), softmax cross-entropy, Adam, Q8_0 block snap, f16 conversion.

**First task (2.0):** port ferrum-cpu's tuned 4x8 CPU matmul into `ferrum_ref` (provenance header, bit-identical-across-threads test) and re-run the Phase 0 comparison against it. From here on every speedup gate uses this tuned baseline, not the simple reference kernel (which is about 2.6x slower than ferrum-cpu's at 2048; Phase 0's MX330 margin at 1024 against the tuned kernel was only 3.02x).

Kernels, one commit each, each with CPU-parity tests on adversarial shapes (`T=1`, batch 1, sizes not divisible by workgroup or 32, large magnitudes): linear+bias, embedding gather, LayerNorm, GELU, residual add, attention (scores/mask/softmax/PV), cross-entropy with `dlogits`, score kernel (log-softmax, target gather, argmax flag). Fuse only where Phase 0's fixed-cost numbers justify it. Tolerances calibrated once and frozen in one constants file.

**Gate:** all parity tests green on every available adapter.

## Status (2026-11-10)

Kernels done and parity-tested on MX330, Intel Xe and llvmpipe: add, bias add, GELU, LayerNorm, causal attention (probs + P·V), cross-entropy with `dlogits` and mean loss, score (log-prob + argmax flag), embedding gather, plus matmul from Phase 0. Worst observed error 3.3e-7, so tolerances are frozen at 1e-5 (`tolerance.rs`). `tests/forward_pass.rs` composes a full 2-block forward pass and loss against the CPU reference: 4.8e-7. Index uploads are range-checked on the host.

**Fusion decision:** none for now. Measured submit+wait is 0.03 ms (MX330) and 0.2 ms (Xe), and each op is one dispatch; fuse only if a Phase 3/4 profile shows dispatch cost dominating.

**2.0 done:** `ferrum_ref::tuned::matmul_tuned` (4x8 register tile, 256x256 panels; bit-identical to the oracle at any thread count). Re-measured: MX330 2.96x at 1024 (just under 3x, within noise) and 4.59x at 2048; Xe 3.51x and 7.15x. See the addendum in `docs/spike-matmul.md`. All later speedup gates use this baseline.

**Phase 2 closed with one caveat:** the forward-pass test composes matmul through host round trips (the elementwise/attention chain is resident). A resident-buffer composition arrives with the Phase 3 model engine.
