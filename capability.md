# What this machine can run (measured), and how that scales

Measured on the machine this repository is developed on, **2026-11-14**. Hand-estimated figures for
other hardware are in [hardware-caps.md](hardware-caps.md); nothing in this file is an estimate except
where it says so.

## The machine

| | |
|---|---|
| OS, compiler | Linux 7.0.0-42-generic (Zorin OS 18), rustc 1.96.0, `wgpu` 30.0.1, release profile with LTO off |
| CPU, RAM | 8 logical cores, 15 GB RAM (about 9 GB free when the benchmarks ran) |
| GPU 1 | NVIDIA GeForce MX330, **2 GB** dedicated (1989 MiB free headless), Vulkan, driver 580.178.04, Pascal |
| GPU 2 | Intel Iris Xe (TGL GT2), integrated: its memory is the machine's RAM, Vulkan, Mesa |
| Largest single buffer | 2047 MiB granted on both adapters (`max binding`) |

## How it was measured

- `cargo bench -p ferrum_wgpu --bench capability` (new): real-size model **shapes** with **random
  weights** (speed and memory depend on the shape, not the values; nothing is downloaded), at f32, Q8_0
  and Q4_0. Single-stream decode: 4 prompt tokens, then the mean over 16 more tokens, after a warm-up pass
  that compiles the pipelines. GPU only (the CPU baseline is in the table further down).
- **Each row ran in its own process.** A first version ran all rows in one process and reported
  "GPU out of memory" for rows that fit: a failed load leaves its partial buffers on the device until the
  process ends, so every later row failed too. The numbers below are the one-process-per-row results.
- One run per row, no repetition, so expect a few percent of noise (the 163M f32 row was 41.2, 41.5 and
  46.0 ms in three runs).
- The shapes are the published layer counts and widths of SmolLM-135M, SmolLM-360M, Qwen2-0.5B,
  TinyLlama-1.1B and Qwen2.5-1.5B (all at most 1.5B parameters, the repository's size limit). They are
  shapes, not those models: a real checkpoint was not loaded, so no quality claim is made.
- Raw output is committed in `docs/logs/capability-2026-11-14-*.txt`; the tables below are copied from it.

## Decode of real-size shapes

### NVIDIA GeForce MX330 (2 GB, discrete)

| model | params | precision | plan MiB | ms/token | tokens/s | note |
|---|---|---|---|---|---|---|
| SmolLM-135M shape | 163M | f32 | 712 | 46.0 | 21.8 | |
| SmolLM-135M shape | 163M | Q8_0 | 343 | 51.6 | 19.4 | |
| SmolLM-135M shape | 163M | Q4_0 | 279 | 40.4 | 24.7 | |
| SmolLM-360M shape | 409M | f32 | 1721 | - | - | failed to load: GPU out of memory |
| SmolLM-360M shape | 409M | Q8_0 | 729 | 84.5 | 11.8 | |
| SmolLM-360M shape | 409M | Q4_0 | 557 | 62.4 | 16.0 | |
| Qwen2-0.5B shape | 630M | f32 | 2454 | - | - | failed to load: GPU out of memory |
| Qwen2-0.5B shape | 630M | Q8_0 | 1099 | 85.2 | 11.7 | |
| Qwen2-0.5B shape | 630M | Q4_0 | 864 | 60.2 | 16.6 | |
| TinyLlama-1.1B shape | 1100M | f32 | 4286 | - | - | failed to load: GPU out of memory |
| TinyLlama-1.1B shape | 1100M | Q8_0 | 1449 | 137.1 | 7.3 | |
| TinyLlama-1.1B shape | 1100M | Q4_0 | 956 | 94.7 | 10.6 | |
| Qwen2.5-1.5B shape | 1777M | f32 | 6893 | - | - | failed to load: GPU out of memory |
| Qwen2.5-1.5B shape | 1777M | Q8_0 | 2661 | - | - | failed to load: GPU out of memory |
| Qwen2.5-1.5B shape | 1777M | Q4_0 | 1925 | - | - | failed to load: GPU out of memory |

### Intel Iris Xe (integrated, shares the 15 GB of RAM)

| model | params | precision | plan MiB | ms/token | tokens/s | note |
|---|---|---|---|---|---|---|
| SmolLM-135M shape | 163M | f32 | 712 | 87.4 | 11.4 | |
| SmolLM-135M shape | 163M | Q8_0 | 343 | 88.5 | 11.3 | |
| SmolLM-135M shape | 163M | Q4_0 | 279 | 70.4 | 14.2 | |
| SmolLM-360M shape | 409M | f32 | 1721 | 157.4 | 6.4 | |
| SmolLM-360M shape | 409M | Q8_0 | 729 | 137.1 | 7.3 | |
| SmolLM-360M shape | 409M | Q4_0 | 557 | 105.8 | 9.5 | |
| Qwen2-0.5B shape | 630M | f32 | 2454 | 176.5 | 5.7 | |
| Qwen2-0.5B shape | 630M | Q8_0 | 1099 | 129.5 | 7.7 | |
| Qwen2-0.5B shape | 630M | Q4_0 | 864 | 100.9 | 9.9 | |
| TinyLlama-1.1B shape | 1100M | f32 | 4286 | - | - | skipped: integrated GPU shares RAM; weights on the host plus the plan exceed the 6064 MiB available |
| TinyLlama-1.1B shape | 1100M | Q8_0 | 1449 | - | - | skipped: integrated GPU shares RAM; weights on the host plus the plan exceed the 6126 MiB available |
| TinyLlama-1.1B shape | 1100M | Q4_0 | 956 | - | - | skipped: integrated GPU shares RAM; weights on the host plus the plan exceed the 6122 MiB available |
| Qwen2.5-1.5B shape | 1777M | all | - | - | - | skipped: needs 10169 MiB of free host RAM for the f32 weights, 10033 MiB available |

How to read it:

- **Fits and runs on the 2 GB MX330:** up to the **1.1B** shape at Q4_0 (10.6 tokens/s) and Q8_0 (7.3
  tokens/s). f32 stops between 163M (712 MiB plan, runs) and 409M (1721 MiB plan, out of memory).
- **Does not fit:** the 1.78B shape even at Q4_0 (a 1925 MiB plan on a card with about 1.9 GB free).
- **Intel Xe** ran everything it was given up to the 630M shape at every precision. The 1.1B and 1.78B
  shapes were **not run** on it: the benchmark refuses to start a shape whose f32 weights (built on the
  host first) plus, for an integrated GPU, the device plan would not fit in the free RAM (about 6 GB at the
  time). That is the benchmark's guard, not a measured limit of the Xe.
- Q4_0 is the fastest precision on both adapters at 360M and up, and f32 the slowest; at 163M all three are
  within noise of one another. Decode here is latency-bound, so a 4x smaller file does not give 4x.

## Smaller shapes, with the CPU for comparison

`cargo bench -p ferrum_wgpu --bench llama`, same day. CPU column: `ferrum_ref`, one thread (how a CPU
runtime runs one token); fine-tuning baseline: 8 threads.

**MX330, decode (ms/token, tokens/s, speed relative to the CPU thread):**

| model | params | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |
|---|---|---|---|---|---|
| small | 0.5M | 0.10 (10116/s) | 2.02 (496/s, 0.05x) | 2.22 (451/s, 0.04x) | 2.05 (488/s, 0.05x) |
| medium | 6.3M | 1.70 (587/s) | 5.04 (199/s, 0.34x) | 6.55 (153/s, 0.26x) | 5.40 (185/s, 0.32x) |
| large | 32.0M | 7.59 (132/s) | 11.14 (90/s, 0.68x) | 13.53 (74/s, 0.56x) | 11.07 (90/s, 0.69x) |

**Intel Xe, decode:**

| model | params | cpu f32 (1 thread) | gpu f32 | gpu Q8_0 | gpu Q4_0 |
|---|---|---|---|---|---|
| small | 0.5M | 0.08 (11969/s) | 6.56 (152/s, 0.01x) | 4.58 (218/s, 0.02x) | 4.13 (242/s, 0.02x) |
| medium | 6.3M | 1.55 (647/s) | 9.36 (107/s, 0.17x) | 9.79 (102/s, 0.16x) | 8.86 (113/s, 0.17x) |
| large | 32.0M | 8.55 (117/s) | 20.20 (50/s, 0.42x) | 21.64 (46/s, 0.40x) | 19.03 (53/s, 0.45x) |

**MX330, fine-tuning step (batch 4 x 32 tokens, f32 AdamW, clip 1.0):**

| model | params | cpu ms/step | gpu ms/step | gpu vs cpu |
|---|---|---|---|---|
| small | 0.5M | 20.9 | 11.3 | 1.84x |
| medium | 6.3M | 172.7 | 44.3 | 3.90x |
| large | 32.0M | 837.0 | 141.4 | 5.92x |

**Intel Xe, fine-tuning step:**

| model | params | cpu ms/step | gpu ms/step | gpu vs cpu |
|---|---|---|---|---|
| small | 0.5M | 41.9 | 28.0 | 1.50x |
| medium | 6.3M | 229.8 | 86.2 | 2.67x |
| large | 32.0M | 1139.4 | 178.6 | 6.38x |

- Fine-tuning is where the GPU wins: 1.5x to 6.4x faster than 8 CPU threads on these two adapters, more
  as the model grows. Decode of one token at a time is **not** faster than one CPU thread on shapes up to
  32M parameters on either adapter.
- Training larger than 32M was **not run**. By the plan, full fine-tuning of the 163M shape needs 2687 MiB,
  which cannot fit the MX330; see the estimate in [hardware-caps.md](hardware-caps.md).

## Findings

1. **The memory budget was too generous, and missing for other GPUs. (Corrected 2026-11-17; fixed in
   Studio phase G1.)** This finding was first worded wrongly: it said the pre-flight had no total, and
   blamed it for the "fits" verdicts in this document's benchmark. Those verdicts came from the
   *benchmark*, which passes no budget at all. The command line did already ask `nvidia-smi`, but it used
   the card's **total** memory (2048 MiB), and the measurements above show that is about 20% too
   optimistic: plans up to 1449 MiB (73% of the 1989 MiB free) loaded, plans from 1721 MiB (86%) failed
   with `GPU out of memory`. Non-NVIDIA GPUs got no budget at all. Since G1 the budget is a stated share
   of the **free** memory for a discrete card (80%, from these measurements, on one card), a share of the
   available RAM for integrated and unified-memory adapters (70%, an assumption not measured on such
   hardware), and `ferrum-gpu info` prints each adapter's budget with where it came from; `--gpu-mem-mb`
   still overrides. The benchmark's own "fits" column still ignores memory and should not be read as a
   capacity claim.
2. **A failed load leaks until the process ends** (see the method note). A long-running server that
   loads a model, fails, and tries a smaller one in the same process may hit this. Not investigated further.
3. **On a 2 GB card the practical decode ceiling is about 1.1B parameters at Q4_0 or Q8_0**, around
   7 to 11 tokens per second. At the 1.5B-class shape the resident size is above the free memory.
4. **The memory formula** `M = params x bytes per weight x 1.10 + 300 MB` predicts the measured plans to
   within -24% to +37% (table in hardware-caps.md); it is optimistic for big-vocabulary small models and
   pessimistic for the smallest shape.

## Decode model fitted to the MX330 runs

- Q4_0 rows, time = floor + weight bytes / bandwidth: floor **33 ms**, effective bandwidth **10.3 GB/s** (4 points).
- Q8_0 rows (3 points): floor 44 ms, effective bandwidth 13.0 GB/s.
- The MX330's memory is 64-bit GDDR5; its published peak is about 56 GB/s (product specification, not measured here).
  So the engine reaches roughly 18% of the peak, plus a fixed cost per token (about 30 layers of small dispatches).

## Reproduce

```bash
export CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
# one (shape, precision) per process, on one adapter
FERRUM_GPU_ADAPTER=MX330 FERRUM_CAP_ONLY=TinyLlama FERRUM_CAP_PREC=Q4_0 \
  cargo bench -p ferrum_wgpu --bench capability
FERRUM_GPU_ADAPTER="Iris(R) Xe" cargo bench -p ferrum_wgpu --bench llama
```

`FERRUM_CAP_ONLY` is a substring of the shape name, `FERRUM_CAP_PREC` one of `f32`, `Q8_0`, `Q4_0`.

## Not run

Apple Silicon, AMD, desktop NVIDIA, Windows, macOS, more than 2 GB of dedicated memory, any real
checkpoint (random weights only), batched decode, long contexts (the plan uses 2048 but only 20 tokens
are generated), and quality or perplexity of any quantised shape.
