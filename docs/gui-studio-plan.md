# Studio plan (version 2, 2026-11-17): a self-contained desktop app, built for small and for very large GPUs

Version 1 of this plan had the GUI run `ferrum-gpu` as a child process. The direction changed the same day:
**the Studio must be self-contained, a desktop application like ferrum-cpu's "SLM Studio", with the changes a GPU
engine needs, and designed to run on very powerful GPUs such as an Apple M5 Max.** This version replaces version 1;
phases 1 to 3 of version 1 are done and kept (they are the page, not the plumbing).

## What ferrum-cpu's Studio is (read from `ferrum-cpu/ferrum_gui`, never modified)

A **Tauri 2** desktop app (`tauri`, `tauri-plugin-dialog`), in its own crate outside the main workspace with its own
lockfile, whose Rust side links the engine **in-process** (no child processes) and exposes it as `#[tauri::command]`
functions; the page is vanilla HTML/CSS/JS under `ui/`, with a strict CSP (`script-src 'self'`, no inline script or
style), twelve tabs, Beginner/Expert mode, a tool-tip on every control, live estimates, native file dialogs, and
progress streamed to the page. Its CLI crate is also a library that the app calls, so there is one implementation of
every command.

## What this repository has now, and what is wrong with it for that goal

- `ferrum_gpu_cli` holds the commands (train, finetune, eval, generate, prepare, tokenize, exports) behind a flag
  parser, and **depends on** `ferrum_gpu_gui` for the engine layer. The GUI therefore cannot call the commands.
- `ferrum_gpu_gui` holds the engine layer (`Backend`, `CpuBackend`, `GpuBackend`), a std-only HTTP server, JSON, and the
  page embedded with `include_str!`. It trains and generates a byte-level Llama-style model only.
- Phases 1 to 3 added the tabbed shell, a jobs API that runs the CLI as a **subprocess**, a read-only file browser, and the
  Datasets and Prepare tabs. The subprocess runner is the part this plan replaces; the jobs API shape (start, poll with
  `since`, stop; output lines with progress events) and all the page code stay.

## Target architecture

```
  ferrum_gpu_studio (Tauri 2 app, excluded from the workspace)      ferrum-gpu (CLI)     ferrum-gpu-server
        |  #[tauri::command] = thin calls                              |  flags -> options      |
        |                                                              |                        |
  ferrum_gpu_gui (the browser/headless front end: std HTTP + the same page)                     |
        |                                                                                       |
        +-------------------------------+---------------------------------+---------------------+
                                        |
                              ferrum_gpu_core   (new library)
                 commands (typed options, Reporter, CancelToken) · Backend engine layer ·
                 Studio API (jobs, files, device info, estimates, GGUF inspection) · hardware & memory budget
                                        |
                  ferrum_wgpu · ferrum_ref · ferrum_model · ferrum_gguf · ferrum_tok
```

- **`ferrum_gpu_core`** (new): everything that does work. Each command is a function taking a typed options struct,
  a `Reporter` (log lines and progress events) and a `CancelToken`, and returning a typed result. The CLI parses flags
  into the same structs, so the CLI's behaviour and tests are unchanged; the Studio fills them from forms.
- **Jobs run as threads in-process.** A device fault is an `Err` (the engine already reports out-of-memory and lost
  devices as values) or a caught panic; either ends the job, not the app. The `.stop` file stays for the CLI; the app
  uses the `CancelToken`.
- **Two transports, one page.** The Tauri app and the HTTP front end both call the same `Studio` object. The page talks
  through `rpc.js`: Tauri `invoke` when it runs in the app, `fetch` to `/api/...` in a browser. Jobs are polled in both
  (one code path, testable under node).
- **`ferrum_gpu_studio`** is a separate crate with its own `Cargo.lock`, like ferrum-cpu's, so the Tauri dependency tree
  and its system libraries (webkit2gtk and friends) stay out of the main workspace and its `cargo deny`. It gets its own
  CI job and its own `cargo deny` run.
- **The page** moves to a single `ui/` tree with relative imports (it works under any base URL, which the Tauri asset
  protocol needs) and a strict CSP. The HTTP front end serves the same files.
- **Kept:** the HTTP front end (headless use over an SSH tunnel, the Chrome-driven tests, the in-browser WebGPU
  runtime). Not planned: a mobile build, auto-update, code signing (documented as steps for the owner).

## Decisions taken while building (2026-11-17)

- **R2 kept the command interface as "argument list in, report out".** Instead of a second typed-options layer that
  could drift from the CLI, the whole command layer moved into `ferrum_gpu_core::commands`; a job calls
  `commands::run(argv, &mut sink)`. The CLI is a three-line `main`; the page builds the same argument lists from its
  forms (`web/tabs/args.js`, tested under node).
- **R3 did not add a second transport or move the page.** The desktop app runs the GUI's loopback server in-process
  and points its window at it, so there is one transport (HTTP), one guard (Host/Origin), and one page. What the app
  needed from the page was a strict Content-Security-Policy, now sent on every static response and tested
  (`tests/csp.rs`). `rpc.js` and a `ui/` move are dropped; native file dialogs are not used (the page has its own
  server-side file picker). Revisit if a native-only feature needs Tauri commands.
- **Cancellation is cooperative.** Stop raises the job's flag; training (both trainers), generation and the byte
  model's loop poll it. `prepare`, `tokenize` and the exports have no check and run to their end; the UI says so.
- **Device-using commands take turns** (adapter enumeration is not safe from several threads); the others run beside them.

## Phases

**Refactor (required first)**

| # | Phase | Deliverable | Tests |
|---|---|---|---|
| R1 | **Extract `ferrum_gpu_core`** (done) | Move the engine layer, JSON, tokenizer, cancel and stop helpers out of the GUI crate into the new library; the GUI, CLI and server depend on it | the moved code's tests move with it; workspace stays green; dependency-direction test |
| R2 | **Commands as library functions** (done) | One command at a time (info, prepare, tokenize, train, finetune, eval, generate, export-*): typed options, `Reporter`, `CancelToken`; the CLI becomes flag parsing plus a call | CLI test suite unchanged and green after each command; new unit tests on the options and on cancellation |
| R3 | **The Studio API and the in-process jobs** (done) | `core::studio` with jobs (threads), files, device info; the HTTP front end becomes a thin transport; the subprocess runner is deleted; `rpc.js` transport abstraction; `ui/` layout, relative imports, CSP | jobs tests re-pointed at in-process commands; node tests for `rpc.js`; the Chrome tests still pass |
| R4 | **The desktop app** (done) | `ferrum_gpu_studio`: Tauri 2 window, commands, native file dialogs, icons, bundle config, CSP; CI job per OS; `cargo deny` for it | app launches headless under a virtual display in a test where possible; command wiring unit-tested without a window |

**Tabs (on the in-process core)**

| # | Phase | Deliverable |
|---|---|---|
| 4 | **Train (GPT-2, BPE)** (done 2026-11-17) | BPE, tie, dropout, QAT, validation share and patience, checkpoint and resume, stop; live loss and validation chart; device, adapter and GPU memory fields |
| 5 | **Fine-tune, Evaluate, Generate** (done 2026-11-17) | fine-tune a GGUF, loss and perplexity, streamed generation with top-k, top-p, repetition penalty |
| 6 | **GGUF and Models** (done 2026-11-17) | inspect a GGUF in-process (header, metadata, tensors, quantisation), list models in a folder, load into the lab |
| 7 | **Export and Serve** | export GGUF (f32, int8, int4), Ollama, LM Studio with tool detection; run the server in-process with port, token and live log |
| 8 | **System and Capable** | adapters, drivers, memory; what fits (the HARDWARE-CAPs.md formulas, implemented in Rust and tested against the script's tables) |
| 9 | **GPU lab** | adapter picker, VRAM budget and live meters, CPU-versus-GPU check, precision comparison, the WebGPU self-test and benchmark, a benchmark report anyone can send back |
| 10 | **Polish and docs** | accessibility pass, empty and error states, `docs/gui.md` rewritten, README |

**GPU scale (so a very powerful GPU is used, not just supported)**

| # | Phase | Why and what |
|---|---|---|
| G1 | **Features and the memory budget** (done 2026-11-17) | The device is already opened with the adapter's own reported limits (the roughly 2 GiB per-buffer cap on this machine's two Vulkan adapters is what their drivers report; another backend may report more, which is not run here), so this phase is about **features** (f16, subgroups, cooperative matrix where `wgpu` exposes them: detected, shown, and requested only when a kernel uses them) and reading memory honestly: unified memory from the operating system (a stated fraction is usable), discrete from the driver where it can be read, always shown as an estimate; the pre-flight uses it, which fixes the finding in CAPABILITY.md that a 4 GB plan "fit" a 2 GB card |
| G2 | **Tensors larger than one binding** | A matrix or embedding bigger than the binding limit is stored as row blocks in several buffers and the kernels run per block (or take several bindings); needed for 70B-class layers and big vocabularies on adapters with smaller limits, and testable here by lowering the limit |
| G3 | **f16 weights, activations and KV cache** | Halves memory and bandwidth where the adapter has f16; the f32 path stays the reference; parity tests against it with stated tolerances |
| G4 | **Kernels that use the hardware** | Tiled matmul with subgroup or cooperative-matrix paths chosen per adapter, fused attention that does not materialise the score matrix, batched prefill and batched decode (decode is latency-bound today, see CAPABILITY.md); each kernel kept only if its benchmark beats the old one on a real adapter |
| G5 | **Verification kit for hardware this machine does not have** | A one-click report (adapter, limits, features, parity tests, decode and prefill speed per precision, largest model that loads) that an owner of an M5 Max or any other machine can run and send back; the rows stay "not run" in the docs until a report exists |

Order: R1, R2, R3, R4, then G1 (the tabs need real capabilities and memory figures), then tabs 4 to 8, G2, tab 9, G3, G4, G5,
tab 10. Merges to `main` follow the gate (fmt, clippy, four test configurations, time limits, coverage floor, `cargo deny`)
after R1, R3, R4, and after each group of tabs.

## What cannot be verified here, said plainly

- **Apple Silicon** (M5 Max or any M-series): this machine is Linux with an NVIDIA MX330 (2 GB) and an Intel Xe. The macOS
  CI runner has a paravirtual GPU that already disagrees with the CPU in one test. Everything for Apple is built to be
  correct by construction and by parity tests against the CPU reference on the adapters available, and carries "not run on
  Apple Silicon" until a G5 report from real hardware exists. Speed and memory figures for such machines stay estimates.
- **Desktop bundles on macOS and Windows** (signing, notarisation, installers): configured, built in CI where the hosted
  runners allow, not tested by hand.
- **Very large models**: no model above 1.5B parameters is downloaded (project rule); large-model behaviour is tested with
  random-weight shapes, as in CAPABILITY.md, and with limits lowered artificially.

## Rules that apply throughout

- Tests first, one concern per file, a `tests/` sibling per module; the suite passes with no GPU, no Chrome, no node.
- No unsafe, no dependency outside this repository except the registry crates the app needs (Tauri and its plugin, in the
  app crate only). Jobs run only the fixed list of commands; paths are values, never shell strings.
- Estimates in the UI say they are estimates; no invented figures in the docs.
- The CLI keeps working exactly as before at every step.
