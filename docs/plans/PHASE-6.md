# Phase 6: own CLI and server

- [x] 6.1 `ferrum_gpu_cli` (binary `ferrum-gpu`): `train`, `generate`, `eval`, `run-gguf`, `finetune-gguf`, `info`; flag-table parser rejecting unknown flags; `--device cpu|gpu|auto`, `--gpu-adapter`, `--threads`. **`--gpu-mem-mb` is deferred to 6.4** (it needs the pre-flight estimate). `eval` runs on the CPU oracle; the tokenizer is the byte tokenizer (BPE is not in the 6.x list).
- [x] 6.2 `ferrum_gpu_server` (binary `ferrum-gpu-server`): `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood` behind a `Decoder` trait; tests use `StubDecoder` and a real-socket test. Loopback only (Host/Origin guard from the GUI HTTP layer); scoring runs on the CPU oracle.
- [ ] 6.3 `auto` rule from the Phase 3/4 crossover measurements, fallback to CPU with one printed line; tested over fake adapter tables.
- [ ] 6.4 Pre-flight memory estimate and refusal.
- [ ] 6.5 E2E through the real binaries on tiny fixtures; tutorial with pasted real output.

**Gate:** E2E green on the dev box.
