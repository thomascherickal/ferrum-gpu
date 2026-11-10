# HANDOFF (updated 2026-11-10 after Phase 4 completed)

> Not yet consumed. A fresh agent: read this first, then add a line here saying when you
> consumed it, and read `CLAUDE.md` (rules 1 to 12 are binding, especially: independence from
> ferrum-cpu, tests for everything, stay autonomous and take the Recommended option, and the
> rule-12 handoff mechanism that produced this file).

## Goal

`ferrum-gpu`: a GPU engine (wgpu / WGSL, any GPU incl. Apple Silicon) that trains, fine-tunes and
runs small causal Transformers with GGUF in and out, mirroring ferrum-cpu **without depending on
it** (code is copied in with provenance headers; ferrum-cpu is never modified). Plan index:
`PLAN.md`; per-phase files `docs/plans/PHASE-0..7.md`. Reports: `docs/spike-matmul.md`,
`docs/decode-crossover.md`, `docs/train-speed.md`, `docs/COVERAGE.md`.

## State of the repo

- Work happens in the git worktree `.claude/worktrees/gpu-port-plan` on branch `plan/gpu-port`
  (pushed to origin). `main` (in `/home/thomas/Projects/ferrum/ferrum-gpu`) was fast-forwarded to
  `80f4caa` (Phases 0 to 3) at the user's request. **Phase 4 (and the rule-12 machinery) is on
  `plan/gpu-port` only.** The user said "merge it" for earlier phases; merge again with
  `git merge --ff-only plan/gpu-port` from the main checkout (leave the worktree first with
  ExitWorktree action keep) when asked. Never push `main` unless the user asks.
- Everything is committed and pushed except possibly the final commit named below; check
  `git status`. Verified before that commit: fmt, clippy `-D warnings`, `cargo deny`, **265 tests
  pass in each of four configurations** (MX330, no GPU, llvmpipe, Intel Xe), coverage **99.33%
  lines (53 missed)**, `FLOOR=99 scripts/coverage.sh` passes.

## Done

- Phase 0 (spike), 1 (device runtime), 2 (forward kernels, tuned CPU baseline), 3 (GGUF, model
  mapping, sampling, GPU decoder; **finding: single-stream decode is slower than one CPU
  thread**, 0.02x to 0.93x), 4 (training: backward kernels, AdamW, loop, GGUF checkpoints,
  micro-batching, **QAT**; **gate met: about 4x to 5x over an 8-thread CPU step**; 2.5x / 1.9x
  ferrum-cpu's real trainer on the reference preset; QAT overhead 0.1% to 2.3%).
- Modular layout (`train/`, `gpt2/` module directories) and rule-11 coverage tooling.
- Rule 12 machinery: `.claude/settings.json` (tracked) runs `scripts/statusline.sh` as the status
  line (records `.superpowers/context.json`, then delegates to the user's own
  `~/.claude/statusline-command.sh`) and `scripts/handoff_check.sh` as `PostToolUse` and
  `UserPromptSubmit` hooks; it injects "write HANDOFF.md now" the first time context use reaches
  98%. Tested (`tests/tests/test_context_scripts.rs`) and dry-run end to end with the exact
  settings commands. Caveats: the status line only refreshes after assistant messages, so a very
  long tool-heavy turn can cross the threshold before the figure updates; headless (`-p`) runs may
  not run the status line; auto-compaction may trigger before 98%, so keep this file current at
  phase boundaries (threshold is `HANDOFF_AT`, default 98).

## Next steps, in order

1. If `git status` shows uncommitted work, run the verification commands below and commit it.
2. **Phase 5** (`docs/plans/PHASE-5.md`): imported Llama/Qwen inference and fine-tuning (RMSNorm,
   RoPE with host-built cos/sin tables for NORM and NEOX plus scaling, grouped-query attention,
   SwiGLU; Q8_0/Q4_0 weights dequantised in-kernel). Copy in (with provenance headers, commit
   `56a9d3a`) the llama tensor mapping from ferrum-cpu `ferrum_core/src/gguf/llama.rs`, `llm.rs`
   and `llm_train/`; generate a tiny synthetic llama fixture with our own GGUF writer; a CPU
   oracle in `ferrum_ref` first, then GPU kernels, each with parity tests and a fault sweep
   (`Gpu::sweep_faults`). Every new file gets unit tests; keep files under ~500 non-test lines.
3. **Phase 6**: our own CLI and server (`--device cpu|gpu|auto`, `--gpu-adapter`,
   `--gpu-mem-mb`), the BPE tokenizer (moved here from Phase 3), the `auto` rule from
   `docs/decode-crossover.md` and `docs/train-speed.md` (single-stream decode stays on the CPU;
   training and batched work go to the GPU), and a real ferrum-cpu-produced GGUF stored as a
   fixture (nothing has loaded one yet: cross-loading is so far verified only against ferrum-cpu's
   source conventions).
4. **Phase 7**: verification matrix `docs/DEVICES.md`, then flip the default to `auto` only for
   device classes with a green row.
5. Weight tying and dropout on the GPU remain open (Phase 4 open items).

## Open questions / decisions already taken

- Taken: wgpu + WGSL; f32 on device; determinism = bit-identical repeat runs on one adapter;
  CPU-vs-GPU by tolerance (`ferrum_wgpu/src/tolerance.rs`: EXACT 1e-6, REDUCING 1e-5, BACKWARD
  1e-5, FORWARD 1e-5, all calibrated against measured errors); default device stays `cpu` until
  Phase 7; QAT snap is bit-exact with the GGUF Q8_0 encoder (the f16 scale uses integer bit
  operations and a literal `+0.0` for zero, because Intel and Mesa compilers keep `-0.0`).
- Not decided: whether to merge or push `main` after Phase 4 (ask only if the user has not said).
- Known gaps: Apple Silicon, AMD and desktop NVIDIA never run; driver versions not recorded; no
  full-length convergence run on real text; speedups vary about 20% run to run on this laptop.

## Commands to run first

```
cd /home/thomas/Projects/ferrum/ferrum-gpu/.claude/worktrees/gpu-port-plan   # or EnterWorktree path=
export CARGO_BUILD_JOBS=3
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads 1                                    # MX330 (default)
FERRUM_GPU_ADAPTER=no-such-adapter cargo test --workspace -- --test-threads 1 # no GPU: must pass
FERRUM_GPU_ADAPTER=llvmpipe cargo test --workspace -- --test-threads 1
FERRUM_GPU_ADAPTER="Iris(R) Xe Graphics (TGL GT2)" cargo test --workspace -- --test-threads 1
FLOOR=99 scripts/coverage.sh
cargo deny check
```

## Gotchas

- **The shell tool refuses "complex" commands inside the worktree** (compound commands with
  pipes plus heredocs, `git -C`, multi-line `sed -i`). Put multi-step work in a script file under
  `$CLAUDE_JOB_DIR/tmp` (`/home/thomas/.claude/jobs/0f74230f/tmp`) and run `bash that.sh`, or use
  the Edit/Write tools. Never write scratch files outside the repo or that directory.
- `cargo fmt --all` reflows code, which breaks later exact-match Edit anchors: re-read the region
  after formatting. Test modules in `train/tests.rs` and `gpt2/tests.rs` are top-level (no `mod`).
- Tests that need a GPU call `open_for_test()` and skip with a printed reason; adapter selection is
  the `FERRUM_GPU_ADAPTER` name fragment. Test-only fault injection: `Gpu::fail_after`,
  `Gpu::sweep_faults` (see `docs/COVERAGE.md`).
- Adapters on the dev box: NVIDIA MX330 (discrete), Intel Iris Xe (Vulkan, integrated), llvmpipe,
  plus Mesa Intel via GL. `ExitWorktree` / `EnterWorktree path=` is how to reach the main checkout.
- Measurements and their caveats live in the `docs/*.md` reports; never quote a number that is not
  in them or freshly measured. Key-bias weights legitimately differ between CPU and GPU (zero true
  gradient, Adam amplifies noise): compare function-space outputs instead.
- ferrum-cpu was run read-only for baselines (binary built into the scratch dir
  `/home/thomas/.claude/jobs/0f74230f/tmp/cpu-target`); ferrum-cpu has an untracked
  `.claude/worktrees/agent-*` from another session that is not ours.
- The user's global `~/.claude/settings.json` was only read, never edited; the project-level
  `.claude/settings.json` overrides the user's `statusLine` for this project and delegates to it.
