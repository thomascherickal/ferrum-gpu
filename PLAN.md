# ferrum-gpu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. ferrum-cpu's working rules apply here too: one helper at a time, tests first, full suite green before every commit.

**Goal:** Train, fine-tune, score and run the same small Transformers as [ferrum-cpu](../ferrum-cpu/README.md) on any GPU (NVIDIA, AMD, Intel, Apple Silicon), with the same GGUF files and the same CLI / server behaviour, while the CPU engine stays the reference.

**Architecture:** One portable backend, `wgpu` with WGSL compute shaders, reaches Vulkan (Linux, Windows, Android), Metal (Apple Silicon and Intel Macs) and DX12 (Windows) from a single safe-Rust code path. The GPU does not get a parallel engine: `ferrum_core` grows two small traits (a training-step engine and a logits decoder), the existing loops (shuffle, batching, checkpoints, stop conditions, metrics, sampling) keep running on the CPU unchanged, and `ferrum_wgpu` implements those traits with device-resident weights. Weights cross between CPU and GPU through the existing `ferrum-checkpoint` and model GGUFs, so no new file format appears.

**Tech Stack:** Rust 2021, `wgpu` (latest stable, pinned in Phase 0), WGSL, `bytemuck` (already in wgpu's tree), `ferrum_core` by path (`../ferrum-cpu/ferrum_core`). No CUDA, no Metal-specific code, no Python.

**Spec:** this file's analysis section, plus `../ferrum-cpu/CLAUDE.md` (hard constraints and working rules). It supersedes the initial sketch (commit `79a32d9`); every decision that sketch left open is resolved in "Decisions" below.

**Status:** plan only, no code yet. Written 2026-11-10 from a read of ferrum-cpu at workspace version 0.3.0.

---

## What ferrum-cpu is (the parts a GPU port touches)

| Area | Where | What the GPU port needs to know |
|---|---|---|
| Tensor | `ferrum_core/src/tensor.rs` | Flat row-major `Vec<f32>` plus shape. f32 everywhere in memory. |
| Kernels | `ops.rs`, `layer/matmul.rs`, `parallel.rs` | Hand-written matmul on a std-only worker pool; bit-identical at any thread count. This is what the GPU replaces. |
| Own stack (GPT-2) | `train_transformer/`, `slm/`, `layer/` | Learned positions, LayerNorm (`eps = 1e-5`), tanh-approximation GELU, dense multi-head attention with a `-1e9` causal mask, optional tied head. Trained from scratch with Adam/AdamW, global-norm clipping, warmup+cosine schedule, FFN dropout, int8 QAT (Q8_0 blocks of 32, f16 scales). |
| Imported stack (Llama/Qwen) | `llm.rs`, `llm_train/`, `gguf/` | RMSNorm, RoPE (NORM and NEOX, with scaling), grouped-query attention, SwiGLU. Inference from f32, Q8_0 or Q4_0 weights; AdamW fine-tuning in f32 only. |
| Training loop | `train_transformer/epoch.rs`, `slm/training_loop.rs` | `forward_train` already runs a whole batch as one `[B*T, C]` matrix. Per step: sync tied head, QAT snap, forward, cross-entropy, backward, restore masters, clip, Adam. The step hook reads only host scalars (loss, step count, lr) except when it writes a checkpoint. |
| Checkpoint | `train_transformer/checkpoint.rs` | GGUF kind `ferrum-checkpoint`: tensors `p{i}.data`, `p{i}.m`, `p{i}.v` (F32) in the canonical order of `params_all()` (token emb, pos emb, 16 params per block, final LN gain and bias, head), plus step, RNG state, epoch start. |
| Decode | `slm/cached.rs`, `llm.rs::generate_inner` | One token in, one `[vocab]` logits row out, KV cache in between; sampling and stop logic on the caller's side. |
| Memory guards | `train_transformer/limits.rs`, `memory.rs`, `slm/memory.rs` | Estimate first, refuse honestly, `--skip` only on explicit request. |
| Front ends | `slm_cli`, `slm_server`, `ferrum_gui` | CLI flags live in per-subcommand `*_FLAGS` tables; the server dispatches through `backend.rs`; the GUI already lists GPUs (`sizing/hardware.rs`: `nvidia-smi` plus sysfs) and reports them as "detected, not used". |

Reference model for every measurement: the 507,552-parameter preset in `../ferrum-cpu/params.yaml` (context 128, embed 96, 4 heads, 3 blocks, hidden 320, vocab 1024).

## Decisions

Each of these was open in the initial sketch. All are taken here; Phase 0 can overturn only the first, and only with measurements.

| Question | Decision | Why |
|---|---|---|
| Backend | `wgpu` + WGSL compute | The only option that covers every vendor and Apple Silicon from one code path with a safe Rust API. CUDA (`cudarc`) is NVIDIA-only, MLX and raw Metal are Apple-only, `ash` is unsafe-heavy. Per-vendor fast paths are not planned; add one only if a measured gap on real hardware justifies its maintenance. |
| Repo shape | New crates in this repo, depending on `ferrum_core` by path | Keeps `ferrum_core` at two dependencies and wasm-clean. ferrum-cpu must never depend on ferrum-gpu (an optional path dependency still has to exist on disk for cargo to resolve, which would break standalone clones of ferrum-cpu). |
| Integration seam | Two traits in `ferrum_core`, implemented here | Reuses every loop, guard and file writer that already exists and is tested. The alternative (a `Backend` trait with one method per op) pushes a readback or an abstraction boundary between every kernel and loses the fusion that makes a GPU worth using. |
| Binaries | GPU-enabled binaries are built here from thin `main.rs` files over the ferrum-cpu front-end libraries | Same reason as repo shape. Names: `train_transformer-gpu`, `ferrum-serve-gpu`. |
| Precision | f32 on device. Q8_0 / Q4_0 weights are dequantised in-kernel for inference. f16 activations are not planned. | `shader-f16` is an optional feature that not every adapter has; f32 runs everywhere. |
| Determinism | Bit-identical across repeat runs on the same adapter, driver, wgpu version and memory budget. CPU-vs-GPU and device-vs-device agreement is by tolerance. | WGSL does not pin the precision of `exp`, `tanh`, `sqrt`, or whether a multiply-add is fused, so cross-device bit equality cannot be promised. ferrum-cpu already scopes its own guarantee to one platform. |
| Default device | `cpu` until the Phase 7 gate passes, then `auto` | No behaviour change for anyone until GPU results are verified. |

## Global Constraints

- GGUF only. No new file kind is needed; the GPU engine reads and writes the existing ones through `ferrum_core`.
- `#![forbid(unsafe_code)]` in every crate root here, including benches and examples. Never call a dependency's `unsafe fn`. Byte casts go through `bytemuck::cast_slice`.
- `ferrum_core` gains no dependency and keeps building for `wasm32-unknown-unknown`. Its existing results stay bit-identical: every seam change in Phase 1 passes ferrum-cpu's determinism test unchanged.
- CPU fallback always works. A missing, lost or too-small GPU produces a clear error or a fall back to CPU (under `auto`), never a panic or an abort.
- No invented numbers. Every figure in docs comes from a real run, recorded with adapter name, driver, backend, OS, wgpu version and date. A platform is claimed only after a green run on it.
- Every dependency added here passes `cargo deny check` against ferrum-cpu's `deny.toml` allow-list (copied into this repo); a licence outside the list is a stop-and-ask, not a silent addition.
- Default branch is `main`. Tests first, tiny models or stubs, one feature per commit, Conventional-style subjects.
- Deliberate shortcuts carry a `// ponytail:` comment naming the ceiling and the upgrade path.
- Low-memory dev box (15 GB RAM, GeForce MX330 with 2 GB VRAM, Intel iGPU): build with `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`, one cargo command at a time.

## Portability rules (what "any GPU or Apple Silicon" requires)

These are design rules, not optimisations. Every kernel and allocation obeys them.

1. **Ask the adapter, assume nothing.** Request the adapter's own limits at device creation and read back what was granted. The WebGPU baseline is small (128 MiB per storage-buffer binding, 256 MiB per buffer, 8 storage buffers per shader stage, 65,535 workgroups per dimension, 256 invocations per workgroup); desktop adapters grant far more, but the code path must be correct at whatever comes back. Confirm the exact field names and defaults against the pinned wgpu version's docs before relying on them.
2. **One buffer per tensor.** No single arena. A tensor larger than the granted binding size is refused with a message naming the tensor and the limit. `// ponytail:` no tensor splitting; add it when a real model hits the limit.
3. **Memory budget is a number we choose, because wgpu does not report VRAM portably.** Sources in order: `--gpu-mem-mb` / `FERRUM_GPU_MEM_MB`; `nvidia-smi` and sysfs (lift the parsers from `ferrum_gui/src/sizing/hardware.rs`); for integrated and unified-memory adapters (Intel iGPU, AMD APU, every Apple Silicon Mac) half of system RAM; otherwise unknown. The override is the calibration knob and always wins.
4. **Unified memory changes the budget, not the code.** On Apple Silicon the GPU shares RAM with the CPU: the host copy of the weights and the device copy count against the same pool, so the estimator adds both when the adapter is integrated.
5. **Allocation failure is an error value.** Every buffer creation in the allocation phase runs inside an out-of-memory error scope and maps to `InferError`, and device loss is caught at the next submit.
6. **Plan, sum, then allocate.** Each engine builds a table of every buffer it will create (name, bytes) before creating any. The estimate shown to the user is the sum of that table, so the estimate and the allocation cannot drift apart.
7. **Shader portability.** WGSL only; f32 and u32/i32 only; no `f16`, no 64-bit integers, no optional features required. Byte-packed quantised weights travel as `u32` arrays. Loops have static or uniform-supplied bounds.
8. **Software adapters are for tests.** llvmpipe/lavapipe, WARP and SwiftShader (device type `Cpu`) are never chosen by `auto`; only an explicit `--gpu-adapter` selects them.
9. **No blocking assumptions.** Readback goes through one helper (`map` + poll), so a later browser/WebGPU build has a single place to make async. Browser support is a non-goal of this plan but must not be designed out.

## Numerical parity rules

Places where a straightforward WGSL translation silently disagrees with the CPU. Each has a named test in the phase that owns the kernel.

- **Rounding.** Rust's `f32::round` rounds halves away from zero; WGSL `round` rounds halves to even. The Q8_0 snap uses `sign(x) * floor(abs(x) + 0.5)`.
- **f16 scales.** QAT snaps to `q * f16(d)`. Reproduce `f32_to_f16` / `f16_to_f32` with integer bit operations (`bitcast`) mirroring `ferrum_core`, not `quantizeToF16`, whose rounding is not guaranteed to match.
- **Q8_0 orientation.** Ferrum `[in, out]` weights are blocked down columns (`fake_quantize_q8_0_cols`); the kernel walks the same stride. Ineligible matrices (`in % 32 != 0`) stay f32, as on CPU.
- **Mask value.** The causal mask is `-1e9`, not `-inf`, and softmax subtracts the row max first, as on CPU.
- **Reduction order.** Every sum (matmul inner product, bias gradient, loss, gradient norm, embedding gradient) is computed by one invocation per output element in ascending index order. No atomics, no two invocations writing one element.
- **Embedding gradient.** Scatter-add with repeated tokens is a race on a GPU. The CPU sorts the batch positions by token and uploads a CSR index (`row_start[vocab + 1]`, `pos[B*T]`); each `(token, column)` output sums its own range in order.
- **Host-computed scalars.** Adam bias corrections (`1 - beta^t` via `powi`), the scheduled learning rate, the clip scale and RoPE angle tables (`RopeScaling::angle`) are computed on the CPU with the existing code and passed as uniforms or uploaded tables, so no `pow`, `sin` or `cos` runs in a shader.
- **Dropout.** Masks come from the CPU RNG (same seed stream as `forward_train`) and are uploaded. `// ponytail:` one upload per step when dropout is on (default is off); move to an in-shader generator only if a profile shows it matters, and accept that CPU parity for dropout runs ends there.
- **Batch split.** When a batch does not fit, it is split into micro-batches whose gradients accumulate on device before one optimizer step. The split is a pure function of shapes and budget, so repeat runs match; a different `--gpu-mem-mb` may change the split and therefore the low bits.

## Review Focus

Failure modes no single kernel test exercises, most likely first. Each is pinned by the named test in the task that owns the code.

1. **A model larger than the device's memory**: expect a refusal before any allocation, naming required and available bytes (`budget_refuses_before_allocating`, Phase 2).
2. **No usable adapter** (headless box, VM, CI without a driver): `--device gpu` errors with the adapter list and the reason; `--device auto` runs on CPU and says so once (`auto_falls_back_to_cpu_without_adapter`, Phase 6).
3. **Repeated tokens in a batch**: the embedding gradient equals the CPU's (`embedding_grad_with_duplicate_tokens_matches_cpu`, Phase 4).
4. **Odd and degenerate shapes**: one-row decode, `T = 1`, batch 1, dimensions not divisible by the workgroup size or by 32 (`kernels_match_cpu_on_adversarial_shapes`, Phase 2 and every kernel after).
5. **Interrupt and resume across devices**: a run stopped on GPU resumes on CPU and the reverse, with step count, RNG and epoch position intact (`checkpoint_round_trips_between_devices`, Phase 4).

---

## File structure

```
ferrum-gpu/
  Cargo.toml                 workspace: ferrum_wgpu, gpu_cli, gpu_server, tests
  deny.toml                  copy of ferrum-cpu's, same allow-list
  rustfmt.toml               copy of ferrum-cpu's
  .github/workflows/ci.yml   lint, test on lavapipe, deny, portable matrix
  ferrum_wgpu/               the engine library
    src/lib.rs               crate root, re-exports
    src/device.rs            adapter listing and choice, granted limits, std-only block_on
    src/budget.rs            budget sources (rule 3), buffer-plan sum, refusal
    src/tensor.rs            GpuTensor: one buffer plus shape; upload, readback
    src/kernel.rs            pipeline and bind-group cache, dispatch-size helper
    src/shaders/*.wgsl       one file per kernel family, included with include_str!
    src/ops/*.rs             one module per kernel family; each owns its CPU-oracle tests
    src/gpt2/infer.rs        Decoder for Ferrum's own models
    src/gpt2/train.rs        StepEngine for TransformerNet
    src/llama/infer.rs       Decoder for imported models (f32, Q8_0, Q4_0)
    src/llama/train.rs       fine-tune engine
    benches/kernels.rs       std-only (harness = false), same style as ferrum-cpu's gemm bench
  gpu_cli/src/main.rs        train_transformer-gpu: slm_cli::run with the GPU provider
  gpu_server/src/main.rs     ferrum-serve-gpu: slm_server with the GPU provider
  tests/tests/*.rs           cross-crate: parity, determinism, no-unsafe scan, e2e on tiny models
  docs/spike-matmul.md       Phase 0 report
  docs/DEVICES.md            the verification matrix (adapter, driver, OS, date, result)
```

Changes in ferrum-cpu are limited to Phase 1 (two traits and their plumbing in `ferrum_core`) and Phase 6 (library entry points in `slm_cli` and `slm_server`). Both are additive.

---

## Phase 0: Spike and go/no-go (1 to 2 days)

**Deliverable:** `docs/spike-matmul.md` with measured numbers and a written decision. Nothing from the spike is kept as product code unless it passes review in Phase 2.

- [ ] **0.1** Create the workspace (`Cargo.toml`, `deny.toml`, `rustfmt.toml`, CI skeleton) and a `ferrum_wgpu` crate with `#![forbid(unsafe_code)]`. Pin the latest stable `wgpu`; record its version and MSRV (set `rust-version` to what it needs; the dev box has rustc 1.96).
- [ ] **0.2** `cargo deny check`. Expected: pass, or a list of licences to raise with the owner before going further.
- [ ] **0.3** `device::list() -> Vec<AdapterSummary>` (name, backend, device type, granted limits). Test: the call returns without panicking when no adapter exists. Run it and paste the dev box's adapters into the report (`vulkaninfo` is not installed there, so this is the first look at what the MX330 and the Intel iGPU report).
- [ ] **0.4** Matmul kernel, three candidates behind one function: (a) one invocation per output element with a scalar inner loop, (b) the same with a `vec4` unrolled inner product, (c) second operand stored transposed so both operands are read contiguously. Each is checked against `ferrum_core::ops::matmul` within tolerance before it is timed.
- [ ] **0.5** Benchmark at 256, 1024 and 2048 squares and at the reference model's real matmul shapes, on every adapter available, against the CPU kernel at 1 and 8 threads. Report three times per case: compute only, compute plus one upload, compute plus upload plus readback.
- [ ] **0.6** Measure fixed costs: one empty dispatch plus submit, and one 4-byte readback. These two numbers decide how much fusion Phase 2 needs and where `auto` should put small-model decode.
- [ ] **0.7** Install lavapipe (`mesa-vulkan-drivers`) and confirm the correctness tests pass on it with the adapter forced by name. This is the CI device.
- [ ] **0.8** Write the report and the decision.

**Gate:** at least 3x the 8-thread CPU kernel at 1024 and above on the MX330 with weights resident (upload excluded, readback included). If it fails, stop and record why. A fail on a 2 GB laptop part is not proof for other hardware, so the report must say whether the bottleneck was compute or fixed overhead.

**Kernel choice is by measurement.** Candidate (a) is the baseline; (b) or (c) is adopted only where it wins on at least two different adapter classes without losing on the third. None of them has been measured yet, and GPU kernel tuning is an area where intuition is unreliable, so no variant is assumed to win.

## Phase 1: The seam in ferrum_core (in ferrum-cpu, additive)

**Deliverable:** `ferrum_core` exposes two traits and the state accessors an engine needs; with no engine supplied every existing result is bit-identical. Follows ferrum-cpu's rules (STATUS.md update, full suite, wasm build).

**Files (ferrum-cpu):** create `ferrum_core/src/engine.rs`; modify `train_transformer/epoch.rs`, `train_transformer/optimizer.rs`, `train_transformer/forward.rs`, `slm/training_loop.rs`, `slm/train*.rs`, `slm/cached.rs`, `slm/generate.rs`, `slm/scoring.rs`, `llm.rs`, `lib.rs`.

**Produces:**

```rust
// ferrum_core::engine

/// What one optimizer step needs from the host.
pub struct StepInput<'a> {
    pub input: &'a [usize],      // B*T token ids
    pub targets: &'a [usize],    // B*T next-token ids
    pub batch: usize,
    pub lr: f32,                 // already scheduled
    pub step: u64,               // 1-based Adam timestep
    pub adam: Adam,              // betas, eps, weight decay
    pub grad_clip: Option<f32>,
    pub dropout: f32,
    pub drop_seed: u64,
    pub want_grad_norm: bool,
}
pub struct StepOutput { pub loss: f32, pub grad_norm: Option<f32> }

/// Runs training steps on its own copy of the weights.
pub trait StepEngine {
    fn load(&mut self, net: &TransformerNet) -> Result<()>;      // host -> device
    fn step(&mut self, s: &StepInput) -> Result<StepOutput>;
    fn sync(&mut self, net: &mut TransformerNet) -> Result<()>;  // device -> host
}

/// Produces next-token logits with an internal KV cache.
pub trait Decoder {
    fn capacity(&self) -> usize;
    fn reset(&mut self);
    fn feed(&mut self, token: usize) -> Result<Vec<f32>>;        // [vocab]
    /// Per position: log-probability of the target and whether it was the argmax.
    fn score(&mut self, tokens: &[usize]) -> Result<Vec<(f32, bool)>>;
}

// impl TransformerNet
pub fn param_state(&self) -> Vec<ParamState<'_>>;   // canonical order; shape, data, m, v
pub fn load_param_state(&mut self, i: usize, data: &[f32], m: &[f32], v: &[f32]) -> Result<()>;
pub fn dropout_mask(rows: usize, hidden: usize, p: f32, seed: u64) -> Vec<f32>;  // the mask forward_train draws
```

- [ ] **1.1** Test `param_state_round_trips`: export, perturb, `load_param_state`, export again, equal; a wrong length is an error. Implement the accessors over `params_all()`.
- [ ] **1.2** Test `engine_none_is_bit_identical`: `steps_over(.., None, ..)` produces the same checkpoint bytes as before the change (golden already in `testdata/pre_epoch_rng.ckpt.gguf`). Thread `Option<&mut dyn StepEngine>` through `steps_over` and `run_training_with`. With an engine the loop builds `StepInput`, calls `step`, runs `check_loss_finite`, advances the step count and step metrics, and calls `sync` before the hook when `checkpoint_every` divides the step, and once when the epoch call returns. Sharding is bypassed when an engine is present.
- [ ] **1.3** Test `fake_engine_drives_the_loop`: a stub engine that wraps the CPU step yields the same weights as the plain path, and its `sync` count equals checkpoints written plus epochs run.
- [ ] **1.4** Add `_with_engine` variants of the config-driven entry points (`train_transformer_config*`, `train_transformer_tokens*`, `train_transformer_from_path*`). Existing functions delegate with `None`. `TransformerConfig` is unchanged.
- [ ] **1.5** Test `cpu_decoder_matches_cached_model`: implement `Decoder` for the existing cached GPT-2 path and for `LlamaModel` + `LlamaCache`; route `generate_*`, `score`, `score_rolling` and `LlamaModel::generate_inner` through `&mut dyn Decoder`. Existing generation and scoring tests pass unchanged.
- [ ] **1.6** Full ferrum-cpu gate: workspace tests, `determinism` test, clippy, fmt, rustdoc, wasm32 build, `cargo deny`. Record the compat note in STATUS.md section 7. Check `../ferrum-ml` still builds.

**Gate:** ferrum-cpu suite green with unchanged counts apart from the new tests.

## Phase 2: Device, buffers, forward kernels

**Deliverable:** `ferrum_wgpu` opens a device, plans and allocates buffers, and runs every forward kernel of the GPT-2 stack with CPU-oracle parity on lavapipe and on the dev box.

**Produces:**

```rust
pub enum DeviceChoice { Cpu, Gpu, Auto }
pub struct GpuOptions { pub adapter: Option<String>, pub mem_mb: Option<u64> }
pub struct Gpu { /* device, queue, granted limits, budget */ }
impl Gpu {
    pub fn list() -> Vec<AdapterSummary>;
    pub fn open(opts: &GpuOptions) -> Result<Gpu>;
    pub fn info(&self) -> &AdapterSummary;
    pub fn budget_bytes(&self) -> Option<u64>;
}
pub struct BufferPlan { /* (name, bytes) rows */ }
impl BufferPlan { pub fn total(&self) -> u64; pub fn check(&self, gpu: &Gpu) -> Result<()>; }
```

- [ ] **2.1** `device.rs`: adapter choice (explicit name substring, else the discrete adapter, else integrated; never a software adapter unless named), granted limits, a std-only `block_on` (`Waker::noop`). Tests: selection order over a fake adapter list; `open` with an unknown name lists what exists.
- [ ] **2.2** `budget.rs`: the sources of portability rule 3, with parsers moved from the GUI. Tests: override wins; integrated adapters use half of RAM; `budget_refuses_before_allocating` (a plan over budget errors with both byte counts and creates no buffer); a tensor over the binding limit is named in the error.
- [ ] **2.3** `tensor.rs` and `kernel.rs`: upload, readback, pipeline cache keyed by shader and entry point, dispatch-count helper that splits across dimensions at the granted workgroup limit. Tests: round trip of odd-length data; dispatch counts at and past the limit; out-of-memory error scope surfaces as `InferError`.
- [ ] **2.4** Forward kernels, one module and one commit each, in this order: matmul (Phase 0 winner) and linear with bias; embedding gather (token + position); LayerNorm; GELU; residual add; causal attention (scores, mask, softmax, then P·V); cross-entropy loss with `dlogits`; score kernel (log-softmax, gather target, argmax flag). Each lands with `kernels_match_cpu_on_adversarial_shapes` for its op: random shapes, `T = 1`, batch 1, sizes not divisible by the workgroup, large-magnitude inputs.
- [ ] **2.5** Fuse where Phase 0's fixed-cost numbers say it pays (candidates: linear + bias + GELU, scores + mask + softmax). One command encoder and one submit per forward pass.
- [ ] **2.6** `benches/kernels.rs` and the first rows of `docs/DEVICES.md`.

**Tolerance:** starting values are relative `1e-5` per kernel and `1e-4` for a whole forward pass. Calibrate once against lavapipe and each real adapter in this phase, then freeze the values in one constants file; a later failure is a bug, not a reason to loosen.

**Gate:** all kernel parity tests green on lavapipe in CI and on both dev-box adapters.

## Phase 3: Inference for Ferrum's own models

**Deliverable:** `gpt2::infer::GpuDecoder` implements `Decoder`; generate, eval and score give CPU-matching results on tiny models.

- [ ] **3.1** Build from a loaded `GenerativeSLM`: walk `model.layers()`, downcast to `Embedding`, `TransformerBlock`, `LayerNorm`, `Linear` (all fields public), upload once. Quantised `Linear`s upload their `QWeight` bytes; f32 ones upload f32.
- [ ] **3.2** KV cache on device: `feed` appends one K and V row per block and reads back one `[vocab]` row. Test `gpu_feed_matches_cpu_feed` over a full context, including the step that fills the cache.
- [ ] **3.3** `score` uses the score kernel so only `T` pairs come back, never `T x vocab` logits. Test against `score_rolling` on CPU.
- [ ] **3.4** Functional test: same prompt, seed and sampling parameters give the same tokens on CPU and GPU for a tiny model at temperature above zero, and the same argmax path at temperature zero except where the top two logits are within tolerance (the test asserts that exception explicitly rather than hiding it).
- [ ] **3.5** Measure tokens per second for decode on CPU and GPU at three model sizes and record the crossover. Small models are expected to decode faster on CPU because each token costs a submit and a readback; the number decides the `auto` rule in Phase 6.

**Gate:** parity tests green; the crossover table is in `docs/DEVICES.md`.

## Phase 4: Training Ferrum's own models

**Deliverable:** `gpt2::train::GpuStepEngine` implements `StepEngine`; a tiny model trains to the CPU's loss curve within tolerance, repeat runs are bit-identical, checkpoints move between devices.

- [ ] **4.1** Backward kernels, one commit each, each checked against the CPU kernel and against finite differences on a tiny case: `dlogits`; linear backward (`dX`, `dW`, `db`); LayerNorm backward; GELU backward; attention backward (`dQ`, `dK`, `dV`); position-embedding gradient; token-embedding gradient through the CSR index (`embedding_grad_with_duplicate_tokens_matches_cpu`); tied-head fold.
- [ ] **4.2** Optimizer: a fixed-order sum-of-squares reduction; one small readback per step returning `[loss, grad_sumsq]`; AdamW kernel taking `lr`, `bc1`, `bc2`, `clip_scale` as uniforms, decaying rank-2 parameters only. Test `adam_step_matches_cpu` over ten steps including a clipped one.
- [ ] **4.3** QAT: masters and snapped copies are separate device buffers, so the CPU's snapshot / restore cycle becomes "snap masters into the snapped buffers, run forward and backward on those, update masters". Tests `q8_0_snap_matches_cpu_bitwise` on ties, on values needing f16 rounding and on ineligible shapes; `qat_model_survives_int8_export_exactly` rerun with GPU-trained weights.
- [ ] **4.4** `load` / `sync` over `param_state`. Test `checkpoint_round_trips_between_devices`: train N steps on GPU, stop, resume on CPU, and the reverse; step count, RNG state and epoch position are preserved, and loss continues within tolerance.
- [ ] **4.5** Micro-batching: when the buffer plan for the requested batch exceeds the budget, halve until it fits, accumulate gradients on device, one optimizer step. Tests: a split batch matches the unsplit one within tolerance; batch size 1 that still does not fit is refused with the estimate.
- [ ] **4.6** Determinism test: two full tiny runs on the same adapter produce byte-identical checkpoints.
- [ ] **4.7** Regression test: the reference preset's first 200 steps on GPU track the CPU loss (`--threads 1`) within tolerance at every step.
- [ ] **4.8** Benchmark a training step for the reference model and a roughly 20M-parameter model, CPU 8 threads against each available adapter.

**Gate:** end-to-end training is at least 2x the 8-thread CPU on the larger model on some real adapter. If only the small model is slower, that is expected and becomes an `auto` rule; if both are, stop and report.

## Phase 5: Imported models (Llama / Qwen)

**Deliverable:** GPU decode and scoring for imported GGUFs in f32, Q8_0 and Q4_0, then f32 fine-tuning.

- [ ] **5.1** Forward kernels with CPU parity: RMSNorm; RoPE from an uploaded cos/sin table built by `RopeScaling::angle` (both NORM and NEOX); grouped-query attention; SiLU and the SwiGLU FFN.
- [ ] **5.2** Quantised linear: Q8_0 and Q4_0 weights packed into `u32` arrays and dequantised in the inner loop, matching `QWeight::dequant_row`. Test against `ops::qlinear`.
- [ ] **5.3** `llama::infer::GpuDecoder` built from `LlamaModel`'s public fields. Tests mirror Phase 3, using the synthetic tiny llama from ferrum-cpu's test support, plus SmolLM2-135M when `.ferrum-dev/models/` has it.
- [ ] **5.4** Fine-tuning: a trait for `LlamaTrainer` shaped like `StepEngine` (added to `ferrum_core::engine` in the same additive way as Phase 1), then backward kernels for RMSNorm, RoPE, GQA and SwiGLU, each against finite differences and the CPU kernel.
- [ ] **5.5** The honest size table: which imported models fit which memory budgets for inference and for fine-tuning, from the buffer plan, checked against at least one real run per row that is claimed.

**Gate:** imported-model generation and scoring match CPU on the tiny llama on every adapter in the matrix.

## Phase 6: Front ends

**Deliverable:** `--device cpu|gpu|auto`, `--gpu-adapter`, `--gpu-mem-mb` in the CLI and server, from binaries built in this repo.

- [ ] **6.1** (ferrum-cpu) `slm_cli`: move the command dispatch behind `pub fn run(argv, engines: &dyn EngineProvider) -> ExitCode` in the library; add the three flags to the relevant `*_FLAGS` tables. The CPU binary passes a provider that answers `--device gpu` with "this build has no GPU support; use train_transformer-gpu". Every existing CLI test passes unchanged.
- [ ] **6.2** (ferrum-cpu) `slm_server`: `bind` takes the same provider; `backend.rs` holds a `Box<dyn Decoder>` per model. `/v1/info` reports device and adapter name.
- [ ] **6.3** `gpu_cli` and `gpu_server`: thin mains supplying the wgpu provider. `info` prints the adapter list, the chosen adapter, granted limits and the budget with its source.
- [ ] **6.4** `auto` rule: GPU when a hardware adapter exists, the buffer plan fits, and the model is above the measured crossover for the task (training and scoring cross over lower than decode); otherwise CPU. One line says which was chosen and why. Tests: `auto_falls_back_to_cpu_without_adapter`; the rule over a table of fake adapters and sizes.
- [ ] **6.5** Memory pre-flight: `train`, `finetune-gguf`, `run-gguf` and the server print the GPU estimate next to the RAM one and refuse on the same terms.
- [ ] **6.6** End-to-end tests through the real binaries on lavapipe: train two epochs, generate, eval, export, reload on CPU, serve and query.
- [ ] **6.7** Tutorial `docs/GPU-QUICKSTART.md`, every command run and its output pasted.

**GUI:** not in this phase. `ferrum_gui` sits outside the ferrum-cpu workspace and carries its own Tauri configuration, so a GPU build means either a second app crate here or a dependency from ferrum-cpu onto this repo, and that choice should be made with the CLI and server already working. Until then the hardware panel keeps saying "detected, not used", which stays true for that binary.

## Phase 7: Verification matrix and default flip

**Deliverable:** `docs/DEVICES.md` filled from real runs; `auto` becomes the default in the `-gpu` binaries.

| Class | Backend | How it gets run | Status |
|---|---|---|---|
| Software (lavapipe) | Vulkan | CI, every push, correctness only | not run |
| NVIDIA discrete (MX330, 2 GB) | Vulkan | dev box | not run |
| Intel integrated | Vulkan | dev box | not run |
| Apple Silicon | Metal | CI macOS arm64 runner for correctness if it exposes a Metal device (to be confirmed by a run); real timings need a physical Mac | not run |
| AMD | Vulkan | needs hardware | not run |
| Windows | DX12 | CI Windows runner (WARP) for correctness if available (to be confirmed); real timings need hardware | not run |

- [ ] **7.1** CI: lint, `cargo deny`, the no-unsafe scan, and the full test suite on lavapipe as the required gate; macOS and Windows as a non-required `portable` job, as in ferrum-cpu.
- [ ] **7.2** Run the suite and the benchmarks on every class that can be reached; record adapter, driver, OS, wgpu version, date.
- [ ] **7.3** Flip the default to `auto` only for classes with a green row. README states exactly which classes are verified and which are expected to work but unrun.

---

## Where the speed comes from

In priority order. The first three are structural and safe to commit to; the rest are adopted only on measurement.

1. Weights, moments, gradients and the KV cache live on the device for the whole run. Per step the host sends token ids and a few scalars and receives eight bytes.
2. One command encoder and one submit per forward-and-backward pass, a second for the optimizer. Pipelines and bind groups are created once.
3. Host sync only at checkpoints and epoch ends (Phase 1's `sync` rule).
4. Fused kernels where fixed dispatch cost dominates (measured in 0.6).
5. Scoring without logits readback (the score kernel).
6. Matmul variant per Phase 0; revisit only with a profile showing matmul still dominates.
7. Micro-batching instead of refusing, when memory rather than compute is the limit.
8. Routing by `auto`, so small models and short decodes stay on the CPU where they are faster.

Not planned: per-vendor backends, f16 or mixed precision, multi-GPU, activation recomputation, browser WebGPU, the GUI. Each is listed so its absence reads as a decision.

## Risks

- **wgpu API churn.** wgpu ships breaking releases several times a year. Pin one version, keep every wgpu call inside `device.rs`, `tensor.rs` and `kernel.rs`, and upgrade deliberately.
- **Driver variance.** Results differ in the low bits across drivers and can differ after a driver update on the same machine. The determinism claim names adapter, driver and wgpu version, and the test compares runs within one process environment.
- **Fixed overhead on small models.** The reference model is tiny; the GPU may lose on it outright. That is an `auto` rule and a documented result, not a failure of the port.
- **2 GB of VRAM locally and no real GPU in CI.** Correctness is covered by lavapipe; performance claims are limited to hardware that was actually run.
- **Apple Silicon is unverified until someone runs it.** The design avoids everything Metal lacks, but ferrum-cpu's own history (two red portable runs that found real bugs) says to expect surprises.
- **Licences in wgpu's tree.** Checked in 0.2 before any real work.

## Execution order

Phase 0 decides whether anything else happens. Phase 1 is in ferrum-cpu and can be reviewed on its own. Phases 2 to 4 are the core of the port and depend on each other in order. Phase 5 and Phase 6 are independent of each other once Phase 4 is green. Phase 7 closes.

Phases 0 to 2 are specified to the task level here. Phases 3 to 7 list tasks, interfaces and named tests, and each gets its step-level plan in `docs/superpowers/plans/` when it starts, because kernel shapes and fusion choices depend on Phase 0's numbers.
