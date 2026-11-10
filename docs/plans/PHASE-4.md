# Phase 4: training

- [ ] 4.1 Backward kernels (each vs CPU reference and finite differences): dlogits, linear, LayerNorm, GELU, attention, position embedding, token embedding via CSR index (duplicate-token test), tied-head fold.
- [ ] 4.2 Fixed-order grad-norm reduction, AdamW kernel with host-computed `lr`, `bc1`, `bc2`, `clip_scale`; one tiny readback per step.
- [ ] 4.3 QAT: master and snapped buffers; Q8_0 snap bit-exact vs reference (ties, f16 rounding, ineligible shapes).
- [ ] 4.4 Own training loop (shuffle without replacement, batching, schedule, stop conditions, metrics) and own `ferrum-checkpoint` GGUF writer/reader; checkpoint fixtures prove resume.
- [ ] 4.5 Micro-batching with on-device gradient accumulation when the batch does not fit.
- [ ] 4.6 Determinism (two runs, same bytes) and loss-curve regression vs the reference trainer.
- [ ] 4.7 Benchmarks on reference and ~20M-parameter models.

**Gate:** curve parity, determinism, and at least 2x the 8-thread CPU on the larger model on some real adapter.
