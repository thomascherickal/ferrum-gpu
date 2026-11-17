# HANDOFF (written 2026-11-17; updated after G2 and Studio tab 9; G3, G4, G5 and tab 10 remain)

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

**Merged to `main`** at `106bdfb`: R1 to R5, G1, Studio tabs 1 to 8 (Datasets, Prepare, Train, Fine-tune, Evaluate, Generate,
Models, Export, Serve, Capable, Device), the desktop app, `scripts/build-apps.sh`, and the corrected plan.

**On branch `worktree-gui-studio`, gate run after these commits** (see `.gate/`; merge to `main` only when it says PASS):
- **G2** (`cdae6da`): tensors larger than one binding are stored as blocks. `ferrum_wgpu/src/models/llama/{shard,embed,lin}.rs`:
  `Lin::Sharded` (column blocks, outputs copied into place), `Embedding::Rows`, `GpuLlama::with_binding_limit` (a lowered limit
  makes a small model shard, which is how it is tested), `plan_limited` and `preflight::decode_buffers_for` (the pre-flight
  lists the blocks). Equality with the whole model is tested for f32, biased, tied, Q8_0, Q4_0, decode and prefill.
  Not sharded: the KV cache, prefill logits and activations. Not run on a real tensor above 2 GiB.
- **`ferrum-gpu lab`** (`06802f5`, `ferrum_gpu_core/src/commands/lab/{mod,parity,speed,report}.rs`): the report that tab 9, G4
  and G5 share. Parity of the GPU against the CPU reference (f32 prefill, decode with KV cache, Q8_0, Q4_0, blocks), decode and
  prefill speed per precision on a random-weight model (`--size small|medium`), adapter, driver (`ferrum_wgpu::device::drivers`),
  features, memory budget, wgpu version (read from `Cargo.lock`), date. `--json` is the file to send back.
- **Tab 9 GPU lab** (`ferrum_gpu_gui/web/tabs/{gpulab,lab_args}.js`, `capableapi/meter.rs`): runs `lab` as a job, JSON report box,
  memory table with the source of each figure (no invented "used" number). Real-Chrome step `gpulab` passes.

## Real numbers from `ferrum-gpu lab --size medium --reps 5` (2026-11-17, release build, wgpu 30.0.1, Linux x86_64)

| adapter, driver | precision | model buffers | decode tok/s | prefill tok/s |
|---|---|---|---|---|
| Intel Iris Xe (TGL GT2), Vulkan, Mesa 25.2.8 | f32 | 99.1 MiB | 58 | 993 |
| | Q8_0 | 31.5 MiB | 53 | 657 |
| | Q4_0 | 19.8 MiB | 64 | 1018 |
| NVIDIA MX330, Vulkan, 580.178.04 | f32 | 99.1 MiB | 98 | 1523 |
| | Q8_0 | 31.5 MiB | 78 | 561 |
| | Q4_0 | 19.8 MiB | 96 | 813 |

CPU reference prefill, f32: 298 and 300 tokens/s. All five parity checks passed on both (errors 1.6e-6 to 3.1e-6 against 1e-5;
blocks 0). **Findings for G4:** Q8_0 prefill is slower than f32 on both adapters (the in-kernel dequantising matmul is
unoptimised); decode is far from bandwidth-bound (8 layers, about 100 dispatches per token). llvmpipe was not run for speed.

## In progress

Nothing uncommitted except possibly this file. Branch `worktree-gui-studio`, worktree `.claude/worktrees/gui-studio`.

## Next steps, in order

1. Confirm the gate (`scripts/gate.sh`, results in `.gate/gate.txt`) says PASS, then fast-forward `main` and push (rule 15).
2. **G3** f16. Start with f16 storage of weights and KV cache with f32 compute (least accuracy risk); gate on the detected
   `shader-f16` feature, request it only when a kernel uses it, fall back to f32. Check what llvmpipe reports first. Xe reports
   f16, the MX330 does not. Parity tests against f32 with stated tolerances; mark deferred f16 activations with `ponytail:`.
3. **G4** kernels: baselines first (the table above is one), keep a kernel only if it beats the old one on a real adapter and
   delete the loser with its numbers documented. Subgroups through `enable subgroups;` should be safe. Cooperative matrix: check
   whether wgpu 30 needs an `unsafe` experimental opt-in (rule 4 forbids it: then record "not done: requires unsafe").
4. **G5** is mostly `lab` plus `docs/verification-kit.md` (how an owner of an M5 Max or other machine runs it and sends the
   file back); Apple rows stay "not run".
5. **Tab 10** polish and docs (accessibility pass, empty and error states, `docs/gui.md` rewrite, README GUI section), last.

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
