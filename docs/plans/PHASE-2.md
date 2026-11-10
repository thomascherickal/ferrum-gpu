# Phase 2: forward kernels + CPU reference

Copy into `ferrum_ref` as needed (provenance headers): LayerNorm (eps 1e-5), tanh-GELU, causal attention (`-1e9` mask), softmax cross-entropy, Adam, Q8_0 block snap, f16 conversion.

**First task (2.0):** port ferrum-cpu's tuned 4x8 CPU matmul into `ferrum_ref` (provenance header, bit-identical-across-threads test) and re-run the Phase 0 comparison against it. From here on every speedup gate uses this tuned baseline, not the simple reference kernel (which is about 2.6x slower than ferrum-cpu's at 2048; Phase 0's MX330 margin at 1024 against the tuned kernel was only 3.02x).

Kernels, one commit each, each with CPU-parity tests on adversarial shapes (`T=1`, batch 1, sizes not divisible by workgroup or 32, large magnitudes): linear+bias, embedding gather, LayerNorm, GELU, residual add, attention (scores/mask/softmax/PV), cross-entropy with `dlogits`, score kernel (log-softmax, target gather, argmax flag). Fuse only where Phase 0's fixed-cost numbers justify it. Tolerances calibrated once and frozen in one constants file.

**Gate:** all parity tests green on every available adapter.
