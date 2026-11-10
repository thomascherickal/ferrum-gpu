# ferrum-gpu plan (index)

**Goal:** train, fine-tune, score and run small causal Transformers on any GPU (NVIDIA, AMD, Intel, Apple Silicon) with GGUF in and out. Behaviour mirrors ferrum-cpu; the code does not depend on it.

**Status:** Phase 0 in progress. Rules: [`CLAUDE.md`](CLAUDE.md). Analysis of ferrum-cpu (read 2026-11-10, workspace 0.3.0) is summarised below; the previous plan revision is in git history (`daf0614`).

## Independence (supersedes the earlier plan)

ferrum-gpu is a standalone repo. There is **no** `ferrum_core` dependency, **no** traits added to ferrum-cpu, and ferrum-cpu is never modified. Everything the GPU engine needs from ferrum-cpu is **copied in** by the phase that needs it, with a provenance header (source path + commit). Consequences:

- The CPU reference (parity oracle) lives in `ferrum_ref/`, a small crate we own. It grows phase by phase: matmul in Phase 0; LayerNorm, GELU, attention, cross-entropy, Adam, Q8_0 and f16 in Phase 2; GGUF, tokenizer and sampling in Phase 3.
- The training loop, checkpoint format and CLI/server are our own. Compatibility is the **GGUF format**, proven with fixture files kept in `fixtures/`.
- ferrum-cpu's own `CLAUDE.md` still mentions this repo; that pointer is now informational only.

## What ferrum-cpu does (what must be reproduced)

- **Own stack (GPT-2):** learned positions, LayerNorm (eps 1e-5), tanh-GELU, dense MHA with `-1e9` causal mask, optional tied head; Adam/AdamW, global-norm clip, warmup+cosine, FFN dropout, int8 QAT = Q8_0 (blocks of 32 down columns, f16 scale, ties away from zero).
- **Imported stack (Llama/Qwen):** RMSNorm, RoPE (NORM/NEOX + scaling), GQA, SwiGLU; f32/Q8_0/Q4_0 inference, f32 AdamW fine-tune.
- **Training step:** whole batch as one `[B*T, C]` matrix: sync tied head, QAT snap, forward, CE, backward, restore masters, clip, Adam. Checkpoint = GGUF `ferrum-checkpoint` with `p{i}.data/m/v` F32 in canonical order.
- **Decode:** one token in, `[vocab]` logits out, KV cache between; sampling on the host.
- **Reference model:** 507,552 params (ctx 128, embed 96, 4 heads, 3 blocks, hidden 320, vocab 1024).

## Decisions

| Question | Decision |
|---|---|
| Backend | `wgpu` (pinned, 30.0.1 at time of writing) + WGSL. One safe-Rust path to Vulkan, Metal, DX12. |
| Precision | f32 on device; Q8_0/Q4_0 dequantised in-kernel for inference. No f16 requirement. |
| Determinism | Bit-identical across repeat runs on the same adapter/driver/wgpu version. CPU-vs-GPU by tolerance. |
| Default device | `cpu` until Phase 7's matrix is green, then `auto`. |
| Layout | Workspace crates: `ferrum_ref` (CPU reference), `ferrum_wgpu` (engine), later `gpu_cli`, `gpu_server`, `tests`. |

## Portability rules

1. Ask the adapter for its limits; never assume more than the WebGPU baseline.
2. One buffer per tensor; refuse (naming tensor and limit) if it exceeds the binding size.
3. Memory budget is chosen by us (wgpu has no portable VRAM query): override > `nvidia-smi`/sysfs > half of RAM on integrated/unified-memory adapters > unknown. Apple Silicon counts host + device copies against one pool.
4. Plan, sum, then allocate: the estimate is the sum of the buffer table.
5. Allocation failure and device loss are error values, never panics.
6. WGSL only, f32/u32/i32 only, no optional features required.
7. Software adapters are chosen only when named explicitly.

## Parity rules

Round-half-away-from-zero in the Q8_0 snap; f16 scale via integer bit ops; `-1e9` mask with max-subtract softmax; every reduction in a fixed ascending order (no atomics); embedding gradient via sorted CSR index; Adam bias corrections, lr, clip scale and RoPE tables computed on the host; dropout masks from the host RNG.

## Phases

| # | File | Deliverable | Gate |
|---|---|---|---|
| 0 | [PHASE-0.md](docs/plans/PHASE-0.md) | Workspace, adapter selection, budget parsers, matmul spike, benchmark | 3x 8-thread CPU matmul at 1024+, else stop |
| 1 | [PHASE-1.md](docs/plans/PHASE-1.md) | Device, buffers, budget, kernel runtime | Parity on tiny ops, suite passes with no GPU |
| 2 | [PHASE-2.md](docs/plans/PHASE-2.md) | Forward kernels + CPU reference | Per-kernel parity on adversarial shapes |
| 3 | [PHASE-3.md](docs/plans/PHASE-3.md) | GGUF I/O, tokenizer, GPT-2 inference | Generate/score match reference |
| 4 | [PHASE-4.md](docs/plans/PHASE-4.md) | Backward, Adam, QAT, training, checkpoints | Loss curve parity, deterministic, 2x on larger model |
| 5 | [PHASE-5.md](docs/plans/PHASE-5.md) | Llama/Qwen inference + fine-tune | Parity on tiny llama |
| 6 | [PHASE-6.md](docs/plans/PHASE-6.md) | Own CLI and server, `--device`, pre-flight | E2E through the binaries |
| 7 | [PHASE-7.md](docs/plans/PHASE-7.md) | Verification matrix, default flip | Only classes with a green row |

Not planned: per-vendor backends, f16/mixed precision, multi-GPU, activation recomputation, browser WebGPU, a GUI.

## Risks

wgpu API churn (keep calls in three files, pin the version); driver variance in low bits; fixed submit/readback cost may make small models faster on CPU (an `auto` rule, not a failure); 2 GB VRAM locally; no real GPU in CI; Apple Silicon unverified until run; licences in wgpu's tree (checked with `cargo deny` in Phase 0).
