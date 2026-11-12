# Test coverage

**After `run`, `--dropout` and the CI fixes (2026-11-12), gate run on the merged result: 686 tests per run in each of four configurations (real GPUs, no GPU, llvmpipe, Intel Xe; 0 failed, 1 ignored each), `FLOOR=98 scripts/coverage.sh` passes: 98.07% of lines (342 of 17743 missed), 96.24% of regions, 93.53% of functions.** The new code is covered by tests split per concern (dropout masks, loss-level gradient checks against finite differences, trainer behaviour including resume, the device rule, the engine choice, the command end to end). `cargo deny`, fmt and clippy `-D warnings` clean. The 99% floor is still not met; this document does not claim it.

**After `--vocab 0`, `--tie` (CPU engine), server cancel on disconnect, `eval-gguf` and `context_len` (2026-11-12), gate run on the merged result: 634 tests per run in each of four configurations (real GPUs, no GPU, llvmpipe, Intel Xe; 0 failed, 1 ignored each), `FLOOR=98 scripts/coverage.sh` passes: 98.00% of lines (350 of 17528 missed), 96.19% of regions, 93.38% of functions.** That is on the floor, not above it: the new code is covered, but the lines it added (the HTTP disconnect watcher's error arms, the tied-engine backend defaults) are mostly paths a real device or socket fault would reach. Same hardware as below; `cargo deny`, fmt and clippy `-D warnings` clean. The 99% floor is still not met; this document does not claim it.

**After M5 (Llama/Qwen run-gguf extras) and the browser-test hardening (2026-11-12), gate run on the merged result: 622 tests per run in each of four configurations (real GPUs, no GPU, llvmpipe, Intel Xe; 0 failed, 1 ignored each), `FLOOR=98 scripts/coverage.sh` passes: 98.06% of lines (336 of 17328 missed), 96.20% of regions, 93.34% of functions.** Hardware was an NVIDIA MX330 and an Intel Iris Xe; `cargo deny`, fmt and clippy `-D warnings` were clean. The 99% floor is still not met; this document does not claim it. The per-file table in `LAST-COVERAGE-RESULTS.md` is from the earlier 98.12% run and was not regenerated.

**After M1 to M7 (2026-11-11), from `scripts/test-all.sh`: 564 tests per run (GPU run and no-GPU run, 0 failed, 1 ignored each), 98.12% of lines, 96.26% of regions, 93.72% of functions.** Lowest files: `ferrum_gpu_gui/src/main.rs` (0%, entry point), `ferrum_gpu_server/src/main.rs` (71.70%, covered by the binary e2e tests, which are not instrumented), `ferrum_gpu_gui/src/engine/gpu.rs` (76.19%: the new GPT-2 GPU engine and decoder arms only run when a hardware adapter is present, and the no-GPU pass cannot reach them), `ferrum_gpu_cli/src/cmds/gpt2/hooks.rs` (87.84%) and `ferrum_model/src/token_file/reader.rs` (88.06%). The per-file table is `LAST-COVERAGE-RESULTS.md`. The 99% floor of earlier runs is not met; this document does not claim it is.

**After Phase 7 (2026-11-11, first measurement that includes the Phase 6 crates `ferrum_gpu_cli` and `ferrum_gpu_server`), from `scripts/test-all.sh`: 432 tests per run, 98.58% of lines (176 of 12415 missed), 96.47% of regions, 95.26% of functions.** This is below the 99% floor the older runs met. The lowest files are `ferrum_gpu_gui/src/main.rs` (0%), `ferrum_gpu_server/src/main.rs` (28.81%, the entry point is exercised by the end-to-end test through the binary, which is not instrumented), `ferrum_gpu_gui/src/engine/gpu.rs` (87.76%) and `ferrum_gpu_cli/src/cmds/mod.rs` (93.87%). The latest per-file table is always in `LAST-COVERAGE-RESULTS.md`; the sections below are older and kept for history.

**After Phases 5a/5b (2026-11-11): 400 tests per run, `FLOOR=99 scripts/coverage.sh` passes: 99.03% of lines (113 of 11638 missed), 96.89% of regions, 95.92% of functions.** The drop from 99.61% is the new `ferrum_gpu_gui` crate: its `main.rs` (a thin entry point) is not covered, and the browser tests (headless Chrome running the page's JavaScript) exercise the server through HTTP but cannot be instrumented, so the JavaScript in `ferrum_gpu_gui/web/` has no line-coverage figure at all; it is checked by the Node and Chrome tests instead. Not re-measured per file.

**Earlier (2026-11-11, after the module restructure and the coverage pass): 355 tests per run; `FLOOR=99 scripts/coverage.sh` passes. llvm-cov summary: 99.61% lines (39 of 9881 missed), 97.20% regions, 96.61% functions.** Reading that honestly: counted per executed line, every line of the workspace now runs (`cargo llvm-cov report --show-missing-lines` lists none). The 39 "missed lines" in the summary belong to 36 functions that never get called, almost all of them error-path closures (`ok_or_else(|| ...)`, `map_err(|e| ...)`) and monomorphised generics, and some are unreachable on a 64-bit target (`usize::try_from(u64)` cannot fail). So the summary is not 100% and this document does not claim it is. About 600 regions (the `?` error arms of parsers and GPU calls) are also unexecuted; region coverage is 97.20%.

Earlier note. Re-measured 2026-11-11 after Phase 5 (349 tests per run; `FLOOR=99 scripts/coverage.sh` passes): **99.35% of lines (67 missed), 96.96% of regions, 96.88% of functions**. The per-file table below is from the 2026-11-10 run (Phase 4) and was not regenerated; the new Phase 5 files (`ferrum_ref/src/llama/*`, `ferrum_model/src/llama_gguf`, `ferrum_wgpu/src/{llama,llama_train,ops_llama,ops_llama_bwd,ops_quant}`) are lower than the older ones in places (e.g. `llama_train/acts.rs` 74.7% lines, `ops_llama.rs` 88.9%, `ops_llama_bwd.rs` 89.5%, `llama/mod.rs` 91.5%): the missed lines are mostly the `?` error paths of allocation and dispatch calls, which only a device fault would reach.

Measured 2026-11-10 with `scripts/coverage.sh` (cargo-llvm-cov, line/region/function coverage).
It merges two full test runs: one on a real GPU (NVIDIA MX330) and one with no GPU, because the
"skip when there is no adapter" branches only execute in the second. 265 tests per run (after QAT and the context scripts were added; `FLOOR=99 scripts/coverage.sh` passes).

**Total: 99.33% of lines (53 missed), 97.50% of regions, 96.37% of functions.**
The floor enforced by the script is whatever `FLOOR=` is set to; it should only be raised to a
number a real run has met (currently 99).

Not measured: the Intel Xe and llvmpipe adapters were not part of this coverage run (they run the
same tests and are exercised for parity, not for coverage); the `benches/` programs are not
instrumented; Windows and macOS were not run.

## Per file

| file | lines | missed |
|---|---|---|
| ferrum_gguf/src/f16.rs | 100.00% | 0 |
| ferrum_gguf/src/quant.rs | 98.86% | 2 |
| ferrum_gguf/src/reader.rs | 97.66% | 9 |
| ferrum_gguf/src/writer.rs | 100.00% | 0 |
| ferrum_model/src/checkpoint.rs | 98.77% | 4 |
| ferrum_model/src/gpt2_gguf.rs | 99.69% | 1 |
| ferrum_model/src/sampling.rs | 99.36% | 1 |
| ferrum_ref/src/gpt2/{decode,mod,params}.rs | 100.00% | 0 |
| ferrum_ref/src/grad.rs | 99.78% | 1 |
| ferrum_ref/src/{matmul,nn,optim,tuned}.rs | 100.00% | 0 |
| ferrum_ref/src/rng.rs | 98.53% | 1 |
| ferrum_ref/src/trainer.rs | 99.67% | 2 |
| ferrum_wgpu/src/{bench_util,budget,select,tolerance}.rs | 100.00% | 0 |
| ferrum_wgpu/src/device.rs | 97.66% | 3 |
| ferrum_wgpu/src/gpt2.rs | 100.00% | 0 |
| ferrum_wgpu/src/matmul.rs | 99.25% | 1 |
| ferrum_wgpu/src/ops.rs | 99.51% | 3 |
| ferrum_wgpu/src/ops_bwd.rs | 99.81% | 1 |
| ferrum_wgpu/src/ops_opt.rs | 99.30% | 2 |
| ferrum_wgpu/src/runtime.rs | 97.48% | 12 |
| ferrum_wgpu/src/tensor.rs | 97.47% | 5 |
| ferrum_wgpu/src/train/{acts,plan,record}.rs | 100.00% | 0 |
| ferrum_wgpu/src/train/mod.rs | 99.66% | 1 |
| ferrum_wgpu/src/train/engine.rs | 96.30% | 1 |

## What the remaining lines are

Most uncovered lines fall into a few kinds, and are not hiding untested features:

1. **Failure-only expressions in test code**, such as the `got[1]` in an assertion message, or the
   `panic!` after a fault sweep that never ends. They run only when a test is already failing.
2. **Shader-compile and pipeline-creation `?` branches.** Reaching them needs a shader that fails
   to compile at the right moment. (The commands issued at run time are covered by fault
   injection, below.)
3. **Polling loops that exit on their first iteration** in the device-lost tests, and the
   single-instruction fallthroughs of defensive `expect`s that cannot trigger (`sampling` after
   float rounding, `rng` rejection retry at probability about 2^-63).
4. **A few reader/tensor error arms** whose inputs need hand-built corrupt GGUF bytes beyond what
   the truncation and hostile-count tests already produce.

## How error paths are covered

GPU code is full of `?` on commands that only fail when the device misbehaves. A test-only hook,
`Gpu::fail_after(n)`, makes the (n+1)th command fail; `Gpu::sweep_faults` runs an operation with
the fault at command 0, 1, 2, ... until it succeeds, asserting each failure reports the injected
fault, leaves no recording batch open, and does not advance state. This is applied to a whole
training step (over 100 commands), to cached decoding (exactly 34 commands for a two-layer model),
to batched prefill and scoring, and to every multi-command kernel wrapper. It also proves a fault
mid-step leaves the weights untouched, because the half-recorded batch is discarded.

## Modularity

Largest non-test file after the split: `ferrum_ref/src/trainer.rs` (about 380 lines).
`train.rs` became `train/{mod,plan,acts,record,engine,tests}.rs` and `gpt2.rs` became
`gpt2/{mod,params,decode,tests}.rs`. Test modules for the large files are siblings
(`tests.rs`), and new files start with their own unit tests (CLAUDE.md rules 1 and 11).

## Next steps toward 100%

- Hand-built corrupt GGUF fixtures for the remaining reader/tensor arms.
- A shader-compile failure hook so pipeline-creation errors can be injected like command faults.
- Restructure assertion messages that index into results so failure-only expressions disappear.
- Raise `FLOOR` after each run that beats it.
