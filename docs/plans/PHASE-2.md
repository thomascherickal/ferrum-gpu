# Phase 2: forward kernels + CPU reference

Copy into `ferrum_ref` as needed (provenance headers): LayerNorm (eps 1e-5), tanh-GELU, causal attention (`-1e9` mask), softmax cross-entropy, Adam, Q8_0 block snap, f16 conversion.

Kernels, one commit each, each with CPU-parity tests on adversarial shapes (`T=1`, batch 1, sizes not divisible by workgroup or 32, large magnitudes): linear+bias, embedding gather, LayerNorm, GELU, residual add, attention (scores/mask/softmax/PV), cross-entropy with `dlogits`, score kernel (log-softmax, target gather, argmax flag). Fuse only where Phase 0's fixed-cost numbers justify it. Tolerances calibrated once and frozen in one constants file.

**Gate:** all parity tests green on every available adapter.
