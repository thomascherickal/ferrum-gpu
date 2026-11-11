# Developing ferrum-gpu

The binding rules are in [CLAUDE.md](../CLAUDE.md); this page explains how to work within them. For the design
see [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Setup

```bash
rustc --version        # 1.87 or newer
cargo install cargo-deny cargo-llvm-cov   # optional but used by the gates
```

The project assumes a **low-memory machine (15 GB)**: use the settings below, and run one cargo command at a
time.

```bash
export CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
```

Work in a git worktree on a branch, never directly on `main`; merge and push to `main` only when the owner
asks. Note that some tooling bases new worktrees on `origin/main` rather than the local `main`; check with
`git log` and `git reset --hard main` if the local `main` is ahead.

## 2. The checks every commit must pass

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check
cargo test --workspace -- --test-threads 1          # real GPU
FERRUM_GPU_ADAPTER=no-such-adapter cargo test --workspace -- --test-threads 1   # no GPU
```

- `--test-threads 1` matters: concurrent GPU device opens can fault some drivers.
- The no-GPU run proves the suite passes on a machine with no adapter; GPU-only tests skip with a printed
  reason. To exercise other adapters, set `FERRUM_GPU_ADAPTER` to a fragment of the adapter's name (for
  example `llvmpipe` or `Xe`).
- One feature per commit, Conventional Commit subjects (`feat(6.2): ...`, `fix: ...`, `docs: ...`).
- The browser tests (`ferrum_gpu_gui/tests/browser_*.rs`) launch headless Chrome in its own process group and
  kill only that group. They skip when Chrome is absent. Never `pkill` a browser by name: the machine also
  runs yours. The slow benchmark test is `#[ignore]`.

## 3. Rules that shape the code

| Rule | In practice |
|---|---|
| Independence | No dependency or file read reaching outside this repository; copy code in with a provenance header (source path and ferrum-cpu commit). `tests/tests/test_independence.rs` enforces it, including `include_str!` paths |
| Tests first, for everything | Every file has unit tests; stubs and fakes stand in for GPUs, `nvidia-smi`, sysfs and Chrome |
| `#![forbid(unsafe_code)]` | In every crate root; never call a dependency's `unsafe fn` |
| No invented numbers | Every figure in the docs comes from a real run, with adapter, driver, backend, OS, `wgpu` version and date; say what was not run |
| Modular files | One concern per file with a small public surface; split at roughly 500 lines of non-test code; big test modules go in a sibling `tests.rs` |
| Deliberate shortcuts | Marked `// ponytail:` with the ceiling and the upgrade path |
| Hardware-only code | Isolated in small files so the rest stays fully testable |

## 4. Coverage

```bash
FLOOR=99 scripts/coverage.sh    # needs cargo-llvm-cov; run alone
```

It merges a real-GPU run with a no-GPU run (the skip branches only run in the second) and fails below
`FLOOR`. Raise the floor only to a figure a real run has met. Current numbers and what they do and do not
mean (the browser JavaScript has no line coverage; the thin `main.rs` files are uncovered) are in
[COVERAGE.md](COVERAGE.md). Phase 6 code has not yet been added to that measurement.

## 5. Benchmarks

Std-only, in `ferrum_wgpu/benches/`:

```bash
cargo bench -p ferrum_wgpu --bench matmul          # Phase 0/2 matmul kernels vs the tuned CPU matmul
cargo bench -p ferrum_wgpu --bench decode          # cached single-token decode, CPU vs GPU
cargo bench -p ferrum_wgpu --bench train           # full training step, CPU vs GPU
cargo bench -p ferrum_wgpu --bench llama           # Llama decode, fine-tune, size table
cargo bench -p ferrum_wgpu --bench profile_train   # where a training step spends its time
```

Rules for reporting: record the adapter, backend, driver if known, OS, `wgpu` version, profile and date; give
medians over several runs; label rows that were not run as plan-only; note run-to-run variance (it is real:
see the QAT section of [train-speed](train-speed.md)). Speed gates compare against the **tuned** CPU matmul
in `ferrum_ref::tuned`, never the simple oracle.

## 6. Adding a GPU kernel

1. **Oracle first.** Write the operation in `ferrum_ref` (host `f32`, plain slices) with unit tests; for a
   backward pass, add a finite-difference check.
2. **Shader.** Add `ferrum_wgpu/src/shaders/<group>/<name>.wgsl` (f32/u32/i32 only, fixed reduction order, no
   atomics) and register it in `shaders.rs`. The registry test checks the list equals the files on disk and
   compiles every entry with `naga`.
3. **Wrapper.** Add a function in the matching `ops/*.rs` that takes resident `GpuBuf`s, looks the pipeline up
   with `pipe!(gpu, "group/name")`, launches with `dispatch_1d` or `dispatch`, and returns errors as values.
4. **Parity test** against the oracle on adversarial shapes (sizes of 1, odd, not multiples of the workgroup
   size, non-square), using a tolerance from `tolerance.rs`. A fault sweep that pins the number of commands
   catches accidental extra dispatches. The test must skip, printing why, when no adapter opens
   (`open_for_test()`).
5. **Plan the buffers** in the engine's `BufferPlan` so the pre-flight sees them.
6. **Determinism.** Run twice and assert identical bits.
7. Do not loosen a tolerance to make a test pass; a failure is a bug.

## 7. Adding a CLI command or server route

- **CLI:** declare the flag table (`val`, `switch`) in `ferrum_gpu_cli/src/cmds/mod.rs`, add the match arm in
  `run`, implement the command as `fn(&Args, &mut dyn Write) -> Result<(), String>`, and test it with the CPU
  backend in `cmds/tests.rs`. Put anything that touches a device behind the `Backend` trait so a fake can
  stand in; use `cmds::gate` for the memory pre-flight. Update `USAGE`, [USER-GUIDE.md](USER-GUIDE.md) and, if
  you have real output, [cli.md](cli.md).
- **Server route:** add the arm in `ferrum_gpu_server/src/routes.rs`, validate every field with a range, return
  `{"error": ...}` with a 4xx status for bad input, and cover it in `routes/tests.rs` against `StubDecoder`.
  Extend the `Decoder` trait only if the route needs the model; implement it in `ModelDecoder` too.
- Keep both loopback-only and bounded (request size, prompt size, token counts).

## 8. Adding a model feature

Add it to the CPU model and the GGUF mapping first (and refuse, rather than approximate, anything you do not
support), then the GPU path with parity tests at every position of a full context. See
[GGUF.md](GGUF.md) for the refusal policy.

## 9. Scripts

| Script | Purpose |
|---|---|
| `scripts/coverage.sh` | Merged GPU and no-GPU line coverage (section 4) |
| `scripts/statusline.sh` | Records the context-window usage from the status-line JSON in `.superpowers/context.json`, then delegates to your own status line (`FERRUM_STATUSLINE_DELEGATE`) |
| `scripts/handoff_check.sh` | Hook that, once usage reaches 98% (`HANDOFF_AT` overrides), tells the agent to write `HANDOFF.md` |

Both context scripts have tests in `tests/tests/test_context_scripts.rs`.

## 10. Hand-offs between sessions

`HANDOFF.md` holds the exact state (what is merged, what is in progress, gotchas); `NEXT-SESSION.md` holds the
start procedure and the remaining plan. A fresh session reads them in that order plus `CLAUDE.md`, then adds a
"consumed" line at the top of `HANDOFF.md`. Keep `HANDOFF.md` current at each phase boundary.

## 11. Documentation standards

- Every public item has a doc comment (`#![warn(missing_docs)]`); `cargo doc --workspace --no-deps` should be
  warning-free.
- Docs state what was and was not run. Do not write a number you did not measure; link the report that holds
  it.
- Keep `PLAN.md`, the phase file and `HANDOFF.md` in step with reality when a phase moves.
