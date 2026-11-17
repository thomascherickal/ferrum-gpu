# HANDOFF (written 2026-11-17; updated after Studio tabs 1 to 8, R1 to R5 and G1)

> Not yet consumed. A fresh agent: read this first, then add a line here saying when you consumed it, and
> read `CLAUDE.md` (the rules are binding: tests first, one feature per commit, the gate before every merge,
> autonomous mode, never touch ferrum-cpu, no invented numbers, rule 16 on process kills). The earlier
> hand-off (Phase 5, consumed 2026-11-11) is `docs/logs/handoff-2026-11-11-phase5.md`.

## The goal

The user asked (2026-11-17): improve the GUI to ferrum-cpu parity with tabs plus GPU-specific extras, then
redirected: **the GUI must be fully self-contained, a desktop app like ferrum-cpu's Tauri "SLM Studio", with the
changes a GPU engine needs, and planned for very powerful GPUs such as the Mac M5 Max.** The plan of record is
**`docs/gui-studio-plan.md` (version 2)**; read it, including "Decisions taken while building" and "What cannot
be verified here". ferrum-cpu is read-only reference (`ferrum-cpu/ferrum_gui`, its Tauri app).

## What is done

**Merged to `main`** (at `d1860ff`): the shell, jobs, the core extraction (R1, R2), in-process jobs and the strict CSP (R3),
the desktop app `ferrum_gpu_studio` (R4), G1 (adapter features and an honest memory budget), the Train tab and the
Fine-tune / Evaluate / Generate tabs, `scripts/gate.sh`, and this hand-off's first version.

**Committed on `worktree-gui-studio`, gate not yet run on them** (run `scripts/gate.sh`, then fast-forward `main`):
- **Models tab** (`modelinfo` in core: inspect any GGUF from its first 64 MiB, type mix, tensors, metadata; folder scan).
- **R5:** the JSON server library moved into `ferrum_gpu_core::serving` (`boot` starts it from the server flags);
  `ferrum_gpu_server` is a thin `main`; `ServeSlot` runs one server in-process.
- **Export and Serve tabs** (jobs on the exports; tool detection through the exports' own `find_tool`; Serve starts and
  stops the server in-process, refuses an open address without a token in the page and in the server).
- **Capable tab** (`ferrum_gpu_core::capacity`: the HARDWARE-CAPs.md formulas in Rust, **a test compares them with
  `scripts/capability-estimates.py`'s table**), `/api/capable`, `/api/system`.
- Every tab is driven in real Chrome by `ferrum_gpu_gui/tests/browser_studio.rs` (`web/autotest/studio.js`): prepare,
  tokenize, train, stop, fine-tune, evaluate, generate, models, export, serve, capable; each tab's pure logic has a node test
  in `ferrum_gpu_gui/js_tests/`; help for every control is checked by `studio_help.rs`.

Earlier in the day (all merged): the details of R1 to R4 and G1 are in `docs/gui-studio-plan.md` ("Decisions taken while
building") and in the commit messages. G1 corrected CAPABILITY.md finding 1 (the CLI had used total VRAM; the "fits"
verdicts came from the benchmark).

## In progress

Nothing uncommitted except possibly this file. Branch `worktree-gui-studio`, worktree `.claude/worktrees/gui-studio`.

## Next steps, in order (from the plan)

1. **Run the gate and merge** the commits listed above as "gate not yet run".
2. **G2** tensors larger than one binding (row-block sharding of a matrix or embedding across several buffers, kernels per
   block; testable here by lowering the limit in `plan.check`). This is what lets 70B-class layers and big vocabularies
   load on adapters with smaller binding limits. **Tab 9 GPU lab** (adapter picker, live memory meters, CPU-versus-GPU
   check, precision comparison, the WebGPU self-test and benchmark moved here, a benchmark report to send back).
   **G3** f16 weights, activations and KV cache (the Intel Xe reports f16; the MX330 does not). **G4** kernels (subgroup
   matmul, fused attention, batched prefill and decode; keep a kernel only if its benchmark beats the old on a real
   adapter; see CAPABILITY.md: decode is latency-bound today). **G5** a verification kit for hardware not available here
   (Apple M-series). **Tab 10** polish and docs (accessibility pass, empty and error states, `docs/gui.md` rewrite).
3. Windows: the desktop app's unit tests do not start there (a Tauri manifest issue, noted in `ci.yml`); fix with a manifest
   for test executables if a Windows machine is available.

## Decisions already taken (do not re-litigate)

Commands stay "argv in, report out" (no second typed layer). One transport (HTTP) for the page and the app. Stop is
cooperative; prepare, tokenize and the exports are not interruptible yet. No native file dialogs (the page has its
own picker). The app crate stays outside the workspace with its own CI job (`studio` in `.github/workflows/ci.yml`,
not required). New markdown files use lowercase kebab-case names (user's global rule); existing uppercase ones stay.

## Commands to run first

```bash
export CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
cargo test --workspace -- --test-threads 1          # add FERRUM_GPU_ADAPTER=no-such-adapter for the no-GPU run
(cd ferrum_gpu_studio && cargo test)                # the app (needs webkit2gtk-4.1, gtk3, libsoup3 on Linux)
scripts/gate.sh                                     # the full gate, about 30 minutes; results in .gate/
scripts/build-studio.sh --debug --run --device cpu  # start the desktop app
```

## Gotchas learned

- **The harness refuses complex compound bash commands** ("too complex to verify that it stays inside the worktree"):
  keep commands simple, do multi-file edits with the Edit tool or a small script file, and never chain `git` with
  other commands in one line. Large inline heredocs of Python sometimes trip it too.
- **The scratchpad directory is cleaned when the session restarts**: keep scripts in the repo (`scripts/`).
- **Stale coverage data**: delete `target/llvm-cov-target` after files move, or coverage reports old paths at 0%.
- **`cargo test` stops at the first failing binary**: a red step can hide later failures; rerun the whole gate.
- **The no-unsafe guard is a text match** (`tests/tests/test_independence.rs`): a test must not contain the word
  as a literal (the CSP test builds the strings with `concat!`).
- **llama.cpp b11521 sometimes prints nothing or a prefix at `-t 1` under load**: the interop helper
  `support/llama.rs::completion` retries (empty or strict prefix only, three tries). Used by the llama.cpp and LM
  Studio tests.
- **Tauri's setup hook runs on the first event-loop iteration**, not at `build()`; mock-runtime windows keep no title.
- A first, all-in-one benchmark process reported false "out of memory" after a failed load: run one shape per process
  (see `CAPABILITY.md`).
- Never signal a process group with a bare negative pid (rule 16); jobs no longer spawn processes at all.
- Apple Silicon, Windows and macOS bundles have not been run by hand; the docs say "not run" and must keep saying it
  until a G5 report from real hardware exists.
