# Phase 6: own CLI and server

- [ ] 6.1 `gpu_cli` (`train`, `generate`, `eval`, `run-gguf`, `finetune-gguf`, `info`) with a flag-table parser rejecting unknown flags; `--device cpu|gpu|auto`, `--gpu-adapter`, `--gpu-mem-mb`.
- [ ] 6.2 `gpu_server` (`/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`) with the decoder behind a trait; tests use a stub decoder.
- [ ] 6.3 `auto` rule from the Phase 3/4 crossover measurements, fallback to CPU with one printed line; tested over fake adapter tables.
- [ ] 6.4 Pre-flight memory estimate and refusal.
- [ ] 6.5 E2E through the real binaries on tiny fixtures; tutorial with pasted real output.

**Gate:** E2E green on the dev box.
