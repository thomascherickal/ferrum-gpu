# HANDOFF (updated 2026-11-11 after Phase 5 tasks 5.0 to 5.3)

> Consumed 2026-11-11 by a Sonnet 5.5 session (read, then continued with Phase 5). Phase 5 progress
> is in the section "Phase 5 status" below; the older text describes Phases 0 to 4.
>
> (Original note:) Not yet consumed. A fresh agent: read this first, then add a line here saying when you
> consumed it, and read `CLAUDE.md` (rules 1 to 12 are binding, especially: independence from
> ferrum-cpu, tests for everything, stay autonomous and take the Recommended option, and the
> rule-12 handoff mechanism that produced this file).

## Phase 5 status (branch `worktree-phase5-llama`, worktree `.claude/worktrees/phase5-llama`, NOT pushed, NOT merged)

Commits on top of `main` (ddfd1ce): 5.0 `12278d1`, 5.1 `13a0760`, 5.2 `a4cdfd3`, 5.3 `c592877`.

- **5.0 done**: `ferrum_ref::llama` (RMSNorm, RoPE NORM/NEOX with host cos/sin tables and
  linear + `rope_freqs` scaling, GQA `attend_row`, SwiGLU, full + KV-cached forward, optional q/k/v
  biases, tied head, `Llama::random`) and `ferrum_model::llama_gguf` (`from_gguf`/`to_gguf`, llama =
  NORM, qwen2 = NEOX, Q8_0/Q4_0, hostile-metadata refusals, scaling policy: none/linear/freq tensor,
  yarn/longrope/attn_factor refused). 29 tests.
- **5.1 done**: `ferrum_wgpu::ops_llama` kernels `rmsnorm`, `rope` (table offset), `swiglu`,
  `gather_rows`, `attention_gqa`, `attention_decode_gqa`; parity vs oracle on MX330/Xe/llvmpipe
  (worst 2.4e-7), fault sweeps pin command counts.
- **5.2 done**: `ferrum_wgpu::ops_quant` (`QKind`, `pack`, `quantise_matrix`, `upload_qweight`,
  `upload_quantised`, `linear_q`, shader `qlinear.wgsl`). Device layout: block-major records
  (`b*n+col`), f32 scale + quants; dequantised values bit-identical to `ferrum_gguf::quant::decode`
  (identity read-back test). encode(decode(x)) is value-idempotent (bytes can differ for blocks
  whose f16 scale underflows). `ferrum_gguf` is now a normal dependency of `ferrum_wgpu`.
- **5.3 done**: `ferrum_wgpu::llama::GpuLlama` (`new(gpu, &Llama, Option<QKind>, Option<Budget>)`,
  `feed`, `forward_all`, `score`, `reset`, `plan`, `dequantised` = the oracle for a quantised run).
  Parity at every position of a full context for 4 variants (llama, qwen+bias+tied, MQA+partial
  rotary, dim 24 with ineligible widths), quantised parity vs `dequantised()`, GGUF round trip.
  Worst error 4.8e-6 against FORWARD 1e-5 (~2x margin; quantised and bias variants are the worst).
  Command counts: decode 40 (f32, 2 layers), `forward_all` 37 for 3 tokens.
  `Llama::random` scales q/k init by `1/sqrt(dim)` (a saturated softmax made parity measure
  conditioning).
- **5.4 done**: CPU oracle backward + AdamW (`ferrum_ref::llama::{loss_and_grads, LlamaTrainer}`, checked against finite differences), GPU backward kernels (`ops_llama_bwd`), `llama_train::GpuLlamaTrainer` (own `step`, `LlamaSnapshot`; tied heads refused). Gradients within 7.6e-7 of the oracle; 12-step loss curve within ~3e-7.
- **5.5 done**: `docs/llama-speed.md` (decode, fine-tune, size table; only shapes up to 32M params were run, larger rows are labelled plan-only). Finding: decode is slower than one CPU thread everywhere; quantised decode is not faster on the GPUs (kernel has no K-split).
- **Verified**: fmt, clippy `-D warnings`, `cargo deny`, 349 tests pass in each of four configurations (MX330, no GPU, llvmpipe, Intel Xe), coverage 99.35% lines (see `docs/COVERAGE.md`).
- **Not done / next**: merge `worktree-phase5-llama` into `main` (user decision; not merged); generalise `Engine`/training loop/checkpoint beyond GPT-2 so Llama fine-tuning can use them; K-split quantised GEMV; Llama tied head, micro-batching, QAT; Phase 6 (CLI/server/BPE, a real ferrum-cpu GGUF fixture) and Phase 7 (device matrix).

## Goal

`ferrum-gpu`: a GPU engine (wgpu / WGSL, any GPU incl. Apple Silicon) that trains, fine-tunes and
runs small causal Transformers with GGUF in and out, mirroring ferrum-cpu **without depending on
it** (code is copied in with provenance headers; ferrum-cpu is never modified). Plan index:
`PLAN.md`; per-phase files `docs/plans/PHASE-0..7.md`. Reports: `docs/spike-matmul.md`,
`docs/decode-crossover.md`, `docs/train-speed.md`, `docs/COVERAGE.md`.

## State of the repo

- At the end of the 2026-11-10 session the user asked for the worktree to be merged and pushed:
  branch `plan/gpu-port` was fast-forwarded into `main` and `main` was pushed to origin, so
  **Phases 0 to 4 and the rule-12 machinery are on `main`**. Verify with
  `git fetch && git log origin/main -1` (the newest commit mentions NEXT-SESSION.md). The old
  worktree `.claude/worktrees/gpu-port-plan` may still exist on disk and can be removed with
  `git worktree remove` once nothing is uncommitted in it. Start new work in a new worktree.
  Never push `main` again without a documented phase gate or an explicit user request.
- Last verified state: fmt, clippy `-D warnings`, `cargo deny`, **265 tests pass in each of four
  configurations** (MX330, no GPU, llvmpipe, Intel Xe), coverage **99.33% lines (53 missed)**,
  `FLOOR=99 scripts/coverage.sh` passes. Start procedure and the full remaining plan are in
  `NEXT-SESSION.md`.

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
