# NEXT-SESSION: how to start, and what to do

Read this first in a fresh session, then `HANDOFF.md` (the exact state and gotchas), then
`CLAUDE.md` (the binding rules). This file is the *start procedure and the plan*; `HANDOFF.md` is
the *state*. After you have read both, add a "consumed" line at the top of `HANDOFF.md`.

## 0. Where things are

- Repository: `/home/thomas/Projects/ferrum/ferrum-gpu`, GitHub `thomascherickal/ferrum-gpu`.
  Work from the **main checkout** (all of Phases 0 to 4 is on `main`). Start any new work in a
  fresh worktree (`EnterWorktree`), never by editing `main` directly.
- Sibling `../ferrum-cpu` is the *reference implementation*. It is read-only for us: never modify
  it, never depend on it (CLAUDE.md "Independence rule"). Copy code in with a provenance header
  that names the source file and ferrum-cpu commit `56a9d3a`.
- Dev hardware: NVIDIA MX330 (2 GB), Intel Iris Xe, llvmpipe (software). No Apple Silicon, AMD or
  desktop NVIDIA available.

## 1. Start checklist (10 minutes)

1. `cd /home/thomas/Projects/ferrum/ferrum-gpu && git fetch && git status -sb && git log --oneline -5`
   Expect `main` equal to `origin/main`, the latest commit mentioning "quantisation-aware training".
2. Check the tools exist: `cargo --version`, `cargo llvm-cov --version`, `cargo deny --version`,
   `jq --version`. Missing ones are a report-and-continue item, not a reason to install with sudo.
3. Prove the baseline is green before changing anything (use a script file under
   `$CLAUDE_JOB_DIR/tmp`, because the shell tool rejects long compound commands):
   ```
   export CARGO_BUILD_JOBS=3
   cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace -- --test-threads 1                                       # MX330
   FERRUM_GPU_ADAPTER=no-such-adapter cargo test --workspace -- --test-threads 1    # no GPU
   FERRUM_GPU_ADAPTER=llvmpipe cargo test --workspace -- --test-threads 1
   FERRUM_GPU_ADAPTER="Iris(R) Xe Graphics (TGL GT2)" cargo test --workspace -- --test-threads 1
   FLOOR=99 scripts/coverage.sh        # needs cargo-llvm-cov; run alone
   cargo deny check
   ```
   Expect 265 tests per run, 0 failures, line coverage about 99.3%. If anything is red, fix that
   first and say so.
4. `EnterWorktree` with a new name (for example `phase5-llama`), so edits are isolated.
5. Confirm the context safety net works: after your first response `.superpowers/context.json`
   should exist (written by `scripts/statusline.sh`). If it does not, the status-line wiring in
   `.claude/settings.json` is not active in this session; keep `HANDOFF.md` current by hand at each
   task boundary.

## 2. Paste-able starting prompt

> Continue ferrum-gpu. Read NEXT-SESSION.md, HANDOFF.md and CLAUDE.md. Run the start checklist,
> then execute Phase 5 (docs/plans/PHASE-5.md) task by task in a new worktree, one commit per
> task, tests first, full suite green before each commit. Stay autonomous: take the Recommended
> option, decide routine questions yourself and say what you decided. Write HANDOFF.md if the
> context-limit hook fires. Do not merge or push main until a phase gate is met and documented.

## 3. The plan (what is left)

Phases 0 to 4 are done and merged. Remaining, in order. Each task ends with: unit tests for every
new file, a CPU parity test, a fault-injection sweep for multi-command ops, a doc update, a
coverage re-measure (`FLOOR=99`), one commit.

### Phase 5: imported Llama / Qwen models (`docs/plans/PHASE-5.md`)

- **5.0 Copy in the llama GGUF mapping.** Read ferrum-cpu `ferrum_core/src/gguf/llama.rs`,
  `llm.rs` (RmsNorm, RoPE NORM/NEOX, GQA attention, SwiGLU, `LlamaModel`) and `llm_train/`
  (backward, AdamW). Write our own `ferrum_ref::llama` (weights type, config, CPU forward and
  KV-cached decode as the oracle) and `ferrum_model::llama_gguf` (tensor names `token_embd`,
  `blk.N.attn_q|k|v|output`, `ffn_gate|up|down`, `output_norm`, `output.weight`, tied head,
  metadata `llama.*`, `rope.freq_base`, `rope.dimension_count`, rope scaling). Generate a tiny
  synthetic llama GGUF with our own writer as the fixture; tests: round trip, quantised round trip,
  hostile metadata, GQA head-count validation.
- **5.1 Forward kernels with parity:** RMSNorm; RoPE from host-built cos/sin tables (never call
  sin/cos in a shader; reuse the host angle code); GQA attention (n_kv_heads divides n_heads;
  scores/softmax/PV over a grouped KV); SiLU and the SwiGLU FFN. Reuse `gemv`, `matmul`,
  `add`, `embed_at`, the batch API and `ops::attention` structure.
- **5.2 Quantised linear:** Q8_0 and Q4_0 weights kept packed on the device (`u32` words) and
  dequantised in the inner loop, matching `ferrum_gguf::quant::decode` bit for bit in the
  dequantised values. This is the lever for memory-bound decode (4x to 8x less traffic).
- **5.3 `GpuLlama` decoder** (feed with a device KV cache, `forward_all`, `score`) mirroring
  `GpuGpt2`; parity against the oracle at every position of a full context; greedy generation
  agreement; budget refusal; fault sweep (count commands exactly).
- **5.4 Llama fine-tuning (f32 AdamW):** backward kernels for RMSNorm, RoPE, GQA, SwiGLU, each
  against finite differences on the CPU oracle first; then a `GpuLlamaTrainer` implementing
  `ferrum_ref::trainer::Engine` so the existing loop, checkpoints, micro-batching and resume work
  unchanged. QAT is optional here.
- **5.5 Size table** from the buffer plans (which imported models fit which budgets for inference
  and fine-tuning), with at least one real run per claimed row.
- **Gate:** tiny-llama parity on every available adapter, GPU fine-tune loss curve matches the CPU
  oracle, and a measured speed comparison (decode and training) written up like
  `docs/train-speed.md`, including the unflattering numbers.

### Phase 6: front ends (`docs/plans/PHASE-6.md`)

Our own CLI (`train`, `generate`, `eval`, `run-gguf`, `finetune-gguf`, `info`; flag tables that
reject unknown flags) and server (`/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`),
`--device cpu|gpu|auto`, `--gpu-adapter`, `--gpu-mem-mb`, the byte-level BPE tokenizer (needs
Unicode category tables: add `unicode-general-category` and `unicode-normalization` only after a
`cargo deny` check), the `auto` rule from the measurements (single-stream decode on CPU, training
and batched scoring on GPU), pre-flight memory estimates, **a real GGUF produced by ferrum-cpu as
a stored fixture** (so cross-loading is proven, not just argued), end-to-end tests through the
real binaries on tiny fixtures, and a tutorial with pasted real output.

### Phase 7: verification matrix (`docs/plans/PHASE-7.md`)

`docs/DEVICES.md` with a row per device class (adapter, driver, OS, wgpu version, date, result or
"not run"), CI (lint, deny, guard tests, suite on llvmpipe; macOS and Windows as non-required
jobs), and the default flip to `auto` only for classes with a green row. Apple Silicon needs a
Mac or a macOS CI runner with a Metal device: say plainly if it was not run.

### Open items carried over

Weight tying and dropout on the GPU; larger-scale convergence run on real text; driver versions
not recorded; remaining coverage gaps listed in `docs/COVERAGE.md` (hand-built corrupt GGUF
fixtures, a shader-compile failure hook, assertion messages that index into results).

## 4. Working rules in one screen (full text in CLAUDE.md)

Independent of ferrum-cpu; tests first and for everything (stubs for external tools and
hardware); one feature per commit; fmt, clippy, deny and the four-configuration suite green
before each commit; no invented numbers (record adapter, driver, date, and say what was not run);
files stay under about 500 non-test lines; take the Recommended option instead of asking; never
force-push; never push `main` without a documented phase gate or an explicit user request; write
`HANDOFF.md` when the context hook fires at 98%.

## 5. Practical gotchas (more in HANDOFF.md)

- Shell tool rejects long compound commands in a worktree: write a script file, run `bash file`.
- `cargo fmt` reflows code, breaking later exact-match edits: re-read after formatting.
- GPU tests skip with a printed reason when no adapter exists; `FERRUM_GPU_ADAPTER` selects one.
- Speedups vary about 20% run to run on this laptop; quote ranges, keep gates generous.
