# Phase 3: GGUF I/O, tokenizer, GPT-2 inference

Copy in (provenance headers): GGUF v3 reader/writer, F32/Q8_0/Q4_0 dequant, byte-level BPE tokenizer, sampling.

- [ ] 3.1 GGUF reader/writer with fixtures in `fixtures/` (tiny gpt2 GGUF generated once, stored in repo).
- [ ] 3.2 Tokenizer and sampling with unit tests.
- [ ] 3.3 `GpuDecoder` for GPT-2 models: weights uploaded once, device KV cache, `feed` returns one `[vocab]` row; `score` via the score kernel.
- [ ] 3.4 Parity: reference decode vs GPU decode over a full context; same seed gives same tokens (argmax ties handled explicitly).
- [ ] 3.5 Measure decode tokens/s on CPU vs GPU at three sizes; record the crossover for `auto`.

**Gate:** parity green; crossover table recorded.
