# Phase 6: own CLI and server

- [x] 6.1 `ferrum_gpu_cli` (binary `ferrum-gpu`): `train`, `generate`, `eval`, `run-gguf`, `finetune-gguf`, `info`; flag-table parser rejecting unknown flags; `--device cpu|gpu|auto`, `--gpu-adapter`, `--threads`. **`--gpu-mem-mb` is deferred to 6.4** (it needs the pre-flight estimate). `eval` runs on the CPU oracle; the tokenizer is the byte tokenizer (BPE is not in the 6.x list).
- [x] 6.2 `ferrum_gpu_server` (binary `ferrum-gpu-server`): `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood` behind a `Decoder` trait; tests use `StubDecoder` and a real-socket test. Loopback only (Host/Origin guard from the GUI HTTP layer); scoring runs on the CPU oracle.
- [x] 6.3 `auto` rule in `ferrum_gpu_cli::device::decide` (also used by the server): training goes to a hardware GPU when one exists (Phase 4: 1.3x to 5.5x), decoding and scoring stay on the CPU (Phase 3: no GPU win up to 34M parameters); no hardware adapter gives the CPU and one printed `note:` line; `--device gpu` and `cpu` are never overridden. Tested over fake adapter tables.
- [x] 6.4 Pre-flight (`ferrum_gpu_cli::preflight`, `cmds::gate`): the same buffer plans the engine checks (`static_plan` + `activation_plan` for training, `llama::plan` for decoding) are checked against the adapter binding limit and `--gpu-mem-mb` before anything is uploaded; a refusal names the adapter, the sizes and the alternatives. Only runs when the backend is a GPU. Budget comes from `--gpu-mem-mb` only (VRAM discovery via `budget::resolve` is not wired in).
- [ ] 6.5 E2E through the real binaries on tiny fixtures; tutorial with pasted real output.

**Gate:** E2E green on the dev box.
