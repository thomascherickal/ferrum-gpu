# Phase 5a: GUI

A local graphical front end for the engine, so a person can see the device, train a small model on text they paste, watch the loss fall, generate text, and move models in and out as GGUF, without a command line.

## Decisions (taken 2026-11-11, autonomous)

| Question | Decision | Why |
|---|---|---|
| Toolkit | A **local web GUI**: a std-only HTTP server in a new crate `ferrum_gpu_gui` serving one HTML page and a JSON API on `127.0.0.1`. | No new dependencies (a native toolkit such as egui adds ~150 crates, needs a display to test, and must pass `cargo deny`); the page is also the host for Phase 5b's in-browser WebGPU. |
| Tokens | A **byte-level tokenizer** (vocab 256). | The real BPE tokenizer is Phase 6. Models trained in the GUI use bytes, so text in and out works end to end today. A GGUF with a different vocabulary loads but the GUI says it can only show token ids. |
| Model | Llama-style (`ferrum_ref::llama`), configurable size presets. | It is the Phase 5 stack; GPT-2 is reachable from the CLI phases later. |
| Devices | `cpu` (the oracle), `gpu`, `auto`. With no adapter the GUI still works on the CPU path. | The suite must pass with no GPU; `--device` semantics arrive here ahead of Phase 6. |
| Bind address | `127.0.0.1` only, random free port by default; no authentication, so never `0.0.0.0`. | It runs code on the user's GPU and reads files by path. |
| Files | Model load/save through the browser (upload / download), never by server-side path from the page. | A page must not be able to make the server read arbitrary paths. |

## Tasks

- [x] 5a.1 `ferrum_gpu_gui` crate: std-only HTTP/1.1 server (request line, headers, `Content-Length` bodies, size limits, timeouts), a small JSON reader/writer, routing; each with unit tests and hostile-input tests (oversized headers, bad lengths, truncated bodies, path traversal on static routes).
- [x] 5a.2 Engine session behind a trait (`Backend`): `info`, `new_model`, `train`, `generate`, `score`, `export`, `import`; a CPU implementation (oracle) and a GPU implementation; tests run both and compare.
- [x] 5a.3 API: `GET /api/info`, `POST /api/model` (preset, seed), `POST /api/train` (text, steps, lr; runs in a background thread), `GET /api/train/status` (loss history), `POST /api/train/stop`, `POST /api/generate`, `GET /api/model.gguf`, `PUT /api/model.gguf`. Errors are JSON with a message, never a panic.
- [x] 5a.4 The page: Device, Train (live loss chart), Generate, Model tabs; accessible labels, no external requests, works at phone width.
- [x] 5a.5 End-to-end: tests drive the real server over TCP; a headless-Chrome run loads the page, trains a few steps and checks the DOM (skipped with a printed reason when Chrome is absent).
- [x] 5a.6 Docs: usage, security notes, screenshot taken from the real run.

**Gate (met 2026-11-11; see `docs/gui.md`):** train, generate and GGUF round trip work through the real server on CPU and GPU with identical API behaviour; the Chrome run passes; suite green with no GPU.

## Not in 5a

Authentication, remote access, multi-user sessions, the real tokenizer (Phase 6), the CLI (Phase 6).

## Outcome notes

- The training text and generation use the byte tokenizer; the GUI holds one model at a time.
- `tests/api.rs` runs the identical user journey on the CPU and the GPU backend (loss 6.7 -> 0.2 in 60 steps on both; the GPU/CPU loss curves differ by at most 4.5e-7 in the engine test), `tests/browser_ui.rs` drives the real page in headless Chrome.
- One crash seen and explained: opening several GPU devices concurrently in one test process segfaulted a driver (SIGSEGV); the repo's rule of `--test-threads 1` avoids it and `docs/gui.md` says so.
