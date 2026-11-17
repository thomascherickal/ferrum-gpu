# HANDOFF (written 2026-11-17; updated after G1 and Studio tabs 4 and 5)

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

Merged to `main` (at `bc2b9c4`): everything in the first list below. **Committed on `worktree-gui-studio` but not yet merged:**
G1 (`d64414a`), Train tab (`b8f870c`), Fine-tune/Evaluate/Generate tabs (`0a0ce6a`); the gate had not been run on them
when this was written (run `scripts/gate.sh`, then fast-forward `main` if green).

Merged first list:

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

## Done since, on the branch (not merged)

- **G1:** `AdapterFeatures` (f16, subgroups, cooperative matrix, timestamps; reported only), `ferrum_wgpu::memory`
  (`resolve_memory`: override, then free VRAM x 80% for a discrete card (measured on the MX330: plans up to 73% of free
  loaded, 86% failed), available RAM x 70% for integrated and unified memory (an assumption), then total x 70%), the
  injected readers in `ferrum_gpu_core::commands::sysmem` (`nvidia-smi` free memory, `/proc/meminfo`, `sysctl`, sysfs),
  `ferrum-gpu info` and `/api/info` and the Device tab show features and the budget with its source. **The earlier
  CAPABILITY.md finding 1 was wrong in its wording and is corrected there** (the CLI did use total VRAM; the "fits" verdicts
  came from the benchmark, which passes no budget).
- **Tab 4 Train** (GPT-2, all the options, live chart, Stop), **tab 5 Fine-tune / Evaluate / Generate**; eval results are
  parsed into an event (`Event::Eval`). The byte-model tabs are now "Lab: model/train/generate", Expert-only (ids kept).
  Page logic is in `web/tabs/*.js` with node tests (`js_tests/*.test.mjs`); `browser_studio.rs` drives the whole chain in real
  Chrome (prepare, tokenize, train, stop, fine-tune, evaluate, generate) against the in-process commands.

## In progress

Nothing uncommitted except possibly this file. Branch `worktree-gui-studio`, worktree `.claude/worktrees/gui-studio`.

## Next steps, in order (from the plan)

1. **Run the gate and merge** the three commits above (see "Done since").
2. **Tabs 6 to 8** on the in-process core: GGUF and Models (inspect a GGUF in-process with `ferrum_gguf`; list models in a
   folder), Export and Serve (`export-gguf`/`export-ollama`/`export-lmstudio` as jobs with tool detection; the server
   in-process), System and Capable (the HARDWARE-CAPs.md formulas in Rust, tested against
   `scripts/capability-estimates.py`; `memory::resolve_memory` already gives the budget). Each tab: its module under `ferrum_gpu_gui/web/tabs/`, its help entries in
   `web/shell/help.js` (the Rust help test enforces them), `args.js` builders with node tests, and a step in
   `web/autotest/studio.js` driven by `browser_studio.rs`. (The byte-model tabs are already relabelled Lab and Expert-only.)
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
