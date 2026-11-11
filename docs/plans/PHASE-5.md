# Phase 5: imported Llama/Qwen models

- [x] 5.1 Kernels with parity: RMSNorm, RoPE (host-built cos/sin tables; NORM and NEOX), GQA, SwiGLU.
- [x] 5.2 Q8_0/Q4_0 linear: weights packed in `u32`, dequantised in the inner loop, vs reference dequant.
- [ ] 5.3 Llama `GpuDecoder` (tiny synthetic llama fixture in repo).
- [ ] 5.4 f32 AdamW fine-tuning: backward kernels vs finite differences.
- [ ] 5.5 Size table from the buffer plan, claimed only where a real run exists.

**Gate:** tiny-llama parity on every adapter in the matrix.
