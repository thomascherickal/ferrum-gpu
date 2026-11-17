# HANDOFF (written 2026-11-17, at the end of Studio phases 1 to 3 and R1 to R4)

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

## What is done and merged (main is at `902350d`; CI for it was queued when this was written)

- **Studio phases 1 to 3:** the tabbed shell (ARIA tabs, Beginner/Expert, theme, a tool-tip on every control with a
  Rust test that every control has help), the Datasets tab and Prepare/Tokenize tab, a read-only file browser, a
  jobs client and panel, node tests (`ferrum_gpu_gui/js_tests`) and real-Chrome tests (`browser_studio.rs`).
- **R1:** `ferrum_gpu_core` extracted (engine layer, `Session`, cancel, http, json, stop, tokenizer).
- **R2:** the whole command layer moved to `ferrum_gpu_core::commands`; the CLI is a three-line `main`; the server
  depends on the core only. Command interface = "argument list in, report to a `Write`".
- **R3:** jobs run **in-process** (`ferrum_gpu_core::jobs`: a thread per job, a line sink, a per-job cancel flag polled
  by both trainers and by generation, device-using commands take turns); the subprocess runner is gone; a strict CSP
  is on every page (`ferrum_gpu_gui/tests/csp.rs`).
- **R4:** `ferrum_gpu_studio/` is the Tauri 2 desktop app, **its own workspace and lockfile**, excluded from the main
  workspace; it runs the GUI's loopback server in-process and shows it in a window. Verified on this machine: it
  launches, serves the page with the CSP, and an X window "Ferrum GPU Studio" 1320x880 appears (see
  `docs/gui.md`). Tests use Tauri's mock runtime (no display). `cargo deny` passes for its tree.
- Gate: all steps green at `0c7016a` (coverage 98.50% lines after clearing a stale `target/llvm-cov-target`).

## In progress

Nothing uncommitted except this hand-off, `scripts/gate.sh` and the archived old hand-off (commit them first).
Branch `worktree-gui-studio`, worktree `.claude/worktrees/gui-studio`.

## Next steps, in order (from the plan)

1. **G1 Features and the memory budget** in `ferrum_wgpu` / core: detect and show adapter features (f16, subgroups,
   cooperative matrix where `wgpu` 30 exposes them); a memory budget that reads honestly (unified memory from the OS
   with a stated usable fraction, discrete from the driver where readable, always labelled an estimate) and is used by
   the pre-flight. Fixes the finding in `CAPABILITY.md` that a plan "fit" a card it did not. Note the device is already
   opened with the adapter's own limits (2047 MiB binding on both Vulkan adapters here).
2. **Tabs 4 to 8** on the in-process core: Train (GPT-2, BPE, live chart), Fine-tune/Evaluate/Generate, GGUF and Models,
   Export and Serve (the server in-process), System and Capable (the HARDWARE-CAPs.md formulas in Rust, tested against
   `scripts/capability-estimates.py`). Each tab: its module under `ferrum_gpu_gui/web/tabs/`, its help entries in
   `web/shell/help.js` (the Rust help test enforces them), `args.js` builders with node tests, and a step in
   `web/autotest/studio.js` driven by `browser_studio.rs`. Relabel the existing byte-model tabs ("Model", "Train",
   "Generate"; ids must stay, the Chrome tests use them) as Lab/Expert-only when the new Train/Generate arrive.
3. **G2** tensors larger than one binding (row-block sharding; testable by lowering the limit), **tab 9 GPU lab**,
   **G3** f16, **G4** kernels (subgroup matmul, fused attention, batching; keep a kernel only if its benchmark beats the
   old on a real adapter), **G5** verification kit for hardware not available here, **tab 10** polish and docs.

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
