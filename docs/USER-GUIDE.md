# User guide

Everything you need to train, run and serve models with ferrum-gpu. Real output for these commands is in
[cli.md](cli.md); how it works inside is in [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Install

You need Rust 1.87 or newer (`rustup`). For GPU use you also need a working driver for Vulkan (Linux,
Windows), Metal (macOS) or DX12 (Windows). Without one everything runs on the CPU.

```bash
git clone git@github.com:thomascherickal/ferrum-gpu.git && cd ferrum-gpu
cargo build --release -p ferrum_gpu_cli -p ferrum_gpu_server -p ferrum_gpu_gui
```

This produces `target/release/ferrum-gpu`, `ferrum-gpu-server` and `ferrum_gpu_gui`. On a machine with
little memory limit the build: `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.

## 2. Concepts you need

- **Model.** `train` has two architectures, chosen with `--arch`:
  - `gpt2` (the default, what ferrum-cpu trains): learned positions, LayerNorm, GELU, **byte-level BPE
    tokenizer stored inside the GGUF**. Sizes come from `--preset` and these flags: `--context --embed --heads
    --blocks --hidden --vocab`. Defaults without a preset: context 16, embed 32, 4 heads, 2 blocks, hidden 64,
    vocab 512, lr 0.01.

    | Preset | Context | Embed | Heads | Blocks | Hidden | Vocab | lr |
    |---|---|---|---|---|---|---|---|
    | `tiny` | 64 | 64 | 4 | 2 | 256 | 512 | 0.01 |
    | `small` | 128 | 128 | 4 | 4 | 512 | 2048 | 0.003 |
    | `medium` | 256 | 256 | 8 | 6 | 1024 | 4096 | 0.001 |

    The vocabulary is a ceiling: if the text has no pair left worth merging, the tokenizer stops short (the
    vocabulary printed in the `model:` line is the real one).
  - `llama` (`--arch llama`): a Llama-style model (RMSNorm, RoPE, GQA, SwiGLU) over **bytes** (vocabulary 256).
    This is also what imported Llama/Qwen files are. Presets: `tiny` 64/32/4:2/2/64, `small` 128/64/4:2/3/176,
    `medium` 128/128/8:2/4/352 (context/dim/heads q:kv/layers/hidden).
- **Tokens.** GPT-2 files use their stored BPE tokenizer (a vocabulary-256 file without one is read as bytes).
  Llama-style files are byte-level. `generate`, `eval` and `finetune-gguf` pick the right path from the file's
  architecture.
- **Device.** `cpu` is the reference implementation (slow, always available). `gpu` needs a hardware adapter.
  `auto` decides per task (section 4).

## 3. Commands

`ferrum-gpu help` prints every flag. Unknown, repeated and valueless flags are errors. `--flag=value` and
`--flag value` both work.

| Command | Purpose | Required | Optional (default) |
|---|---|---|---|
| `info` | Show the device that would be used, every adapter, and optionally a model's shape | | `--model F.gguf` |
| `train` | Train a fresh model and write a GGUF | `--data FILE`, `--out F.gguf` | see below |
| `finetune-gguf` | Continue training a GGUF model (GPT-2 files keep their tokenizer) | `--model`, `--data`, `--out` | the training flags |
| `generate` | Continue a prompt with sampled text | `--model F.gguf`, `--prompt TEXT` | `--max-new` (64), `--temperature` (0 = greedy), `--top-k` (0), `--seed` (1); GPT-2 files also `--top-p`, `--repetition-penalty`, `--stop S` (repeatable), `--stream`, `--ids "1 2 3"` |
| `run-gguf` | Same as `generate`, and first prints the device and the model's shape | as `generate` | as `generate` |
| `eval` | Mean next-token loss and perplexity of a file under a model | `--model`, `--data` | `--ctx N`, `--json F` |

**`train` flags (GPT-2, the default).** Sizes: the preset flags above. How long: `--epochs` (100), or
`--steps N`, or `--tokens N` (a token budget; steps = tokens / (batch x context)). Optimiser: `--lr` (preset's),
`--batch` (16), `--seed` (1337), `--weight_decay` (0), `--clip` (0 = off), `--cosine` with `--warmup N`, `--stride`
(0 = non-overlapping windows). Tokenizer: `--tokenizer F` loads the BPE tokenizer file if it exists, else trains
one and saves it there; without the flag one is trained and kept only inside the model. Output: `--quant
f32|int8|int4` (default int8; `q8_0`/`q4_0` also accepted), `--no-qat` (plain f32 training; QAT is on by
default, as in ferrum-cpu). Safety and control: `--checkpoint F` with `--checkpoint-every N` (a checkpoint is
always written at the end), `--resume F` (refuses a different run shape or tokenizer), `--val X` holds out the
last fraction X of the tokens and reports the validation loss after every epoch, keeping the best epoch;
`--patience N` stops after N epochs without improvement; `--max-seconds X` and a `<out>.stop` file (create it
to stop cleanly after the current step); `--metrics-csv F`; `--verbose` (progress every 10 steps with tokens/s,
gradient norm and ETA); `--sample` (print a short greedy sample at the end).

Not supported yet and refused by name: `--tie` (tied output head), `--dropout` above 0, `--vocab 0`
(character-level). `--force` is accepted and does nothing (there is no RAM guard).

**`train --arch llama` flags.** `--preset`, `--steps` (100), `--lr` (0.003), `--batch` (4), `--context`
(min(model ctx, 32)), `--seed` (1), `--quant f32|q8_0|q4_0` (f32). The GPT-2-only flags above are refused with
`needs --arch gpt2`.

Device flags for `info`, `train`, `finetune-gguf`, `generate` and `run-gguf`: `--device cpu|gpu|auto`
(default `auto`), `--gpu-adapter NAME` (case-insensitive fragment of the adapter name), `--gpu-mem-mb N`,
`--threads N` (CPU threads; default all). `eval` always runs on the CPU.

Notes:

- `generate` prints the prompt followed by the new tokens.
- Greedy generation (`--temperature 0`) is deterministic. With temperature above 0, the same `--seed` gives
  the same text.
- `--quant` quantises the file that is **written**; training itself is f32. Q8_0 and Q4_0 apply to matrices
  whose input dimension is a multiple of 32; the rest stay f32.
- `eval` on the training file measures fit, not generalisation. Hold out text for a real estimate.
- Limits: training text and prompts are read whole; the training `--context` must not exceed the model's
  context; `--steps` is 1 to 100000 and `--batch` 1 to 256.

## 4. Choosing the device

`--device auto` uses the measured crossovers:

- **Training** runs on a hardware GPU if one exists. Software adapters (llvmpipe, SwiftShader) are never
  picked unprompted, because they are slower than the CPU path.
- **Decoding, generation and serving** run on the CPU. On the two GPUs measured, single-token decode was not
  faster than one CPU thread up to 34M parameters. Use `--device gpu` to override (for example for a larger
  model you have measured yourself).
- If `auto` ends on the CPU because no usable GPU exists, it prints one `note:` line saying why.

`--device gpu` fails with a clear message if no adapter matches. To pick one of several, pass
`--gpu-adapter` with part of its name as shown by `ferrum-gpu info`, for example `--gpu-adapter MX330`. The
environment variable `FERRUM_GPU_ADAPTER` selects the adapter for the test suite and tools.

## 5. Memory and the pre-flight

On a GPU run, the buffers are planned and added up **before anything is uploaded**: weights, gradients and
both Adam moments for training, plus the activations of one step; or weights, KV cache and scratch for
decoding. A buffer over the adapter's binding limit, or a total over `--gpu-mem-mb`, is refused with a message
that names the adapter and the sizes and tells you what to change (smaller preset, batch or context; a larger
`--gpu-mem-mb`; or `--device cpu`).

Without `--gpu-mem-mb` the budget is unknown, so only the binding limit is enforced; the machine may still run
out of device memory on a large model, which the engine reports as an error value instead of crashing. Set
`--gpu-mem-mb` to your card's free memory for a firm guarantee.

## 6. Fine-tuning a GGUF model

```bash
ferrum-gpu finetune-gguf --model base.gguf --data new.txt --out tuned.gguf --steps 200 --lr 0.001 --quant q8_0
```

Supported inputs are `llama` and `qwen2` architecture files (see [GGUF.md](GGUF.md) for exactly what loads and
what is refused). The Llama trainer is f32 AdamW and does not support tied output heads: a model whose head is
tied to the embedding is refused. The CLI trains on raw bytes (ids 0 to 255), so it only makes sense for models whose vocabulary covers
them; the vocabulary is otherwise not checked against your data's tokenizer, because the CLI has none.

## 7. The server

```bash
ferrum-gpu-server --model m.gguf [--host 127.0.0.1] [--port 8080] [--token T] [--device cpu|gpu|auto] [--gpu-adapter NAME] [--gpu-mem-mb N]
```

It serves GPT-2 files (with their BPE tokenizer) and Llama-style byte-level files, prints the URL, and serves
calls one at a time per model. The routes, request shapes and limits are ferrum-cpu's `ferrum-serve`, so clients
written for it work.

- **Who can connect.** The default is loopback only: requests whose `Host` is not that address, or that carry a
  cross-site `Origin`, get 403. A non-loopback `--host` is refused at startup unless you set a token (`--token`,
  or the `FERRUM_TOKEN` environment variable) or pass `--allow-unauthenticated`. With a token, every route except
  `GET /health` needs `Authorization: Bearer <token>`.
- **Limits** (`--max-batch` 64 requests, `--max-chars` 65536, `--max-body` 8 MiB, `--max-tokens` 4096) are checked
  before any model work; 32 POSTs may be in flight, more get 503.

| Route | Request | Response |
|---|---|---|
| `GET /health` | | `{"status":"ok"}` |
| `GET /v1/info` | | `{"device","kind","arch","params","vocab","ctx"}` |
| `POST /v1/generate` | `{"requests":[{"prompt", "max_tokens" (64), "stop": [str], "temperature" (0), "top_k" (0), "top_p" (1), "seed" (0), "repetition_penalty" (server default 1)}]}` | `{"results":[{"text","n_tokens"}]}` |
| `POST /v1/loglikelihood` | `{"requests":[{"context","continuation"}]}` | `{"results":[{"logprob","n_tokens","greedy"}]}` |
| `POST /v1/loglikelihood_rolling` | `{"texts":[str]}` | `{"results":[{"logprob","n_tokens"}]}` |

`logprob` is the sum of the log-probabilities of the scored tokens; `greedy` says whether each was the model's
top choice. Unknown fields and out-of-range values are 400 with `{"error": "..."}`; a model error is 422. At most 16
stop strings of 1024 bytes. `top_p` and `repetition_penalty` need a GPT-2 file. Scoring runs on the CPU.

```bash
curl -s -H 'Authorization: Bearer s3cret' -d '{"requests":[{"prompt":"the c","max_tokens":16}]}' http://127.0.0.1:8080/v1/generate
```

## 8. The GUI

`ferrum_gpu_gui` serves a local page to create, train, run and export models and to run inference in the
browser on WebGPU. See [gui.md](gui.md) and [browser-webgpu.md](browser-webgpu.md). Its flags are
`--port`, `--device`, `--gpu-adapter` and `--threads`. Like the server it is loopback-only and
unauthenticated.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `libEGL warning: ... failed to create dri2 screen` on every run | The NVIDIA/Mesa driver stack writes these to stderr while `wgpu` enumerates adapters. Harmless; redirect stderr (`2>/dev/null`) to hide them. |
| `auto` always says "using the CPU for decoding" | That is the rule, not a fault ([section 4](#4-choosing-the-device)). Force with `--device gpu`. |
| `only software or OpenGL adapters found; name one explicitly` | The machine has only llvmpipe-style or OpenGL (Gl) adapters; `auto` and `--device gpu` skip both unless named. Use `--device cpu`, or name the software adapter with `--gpu-adapter` knowing it is slow. |
| `no adapter matches 'X'; available: ...` | The fragment did not match. Copy part of a name from `ferrum-gpu info`. |
| `refusing to start on <GPU>: needs N bytes ...` | The pre-flight refused the run. Follow the advice in the message. |
| `the loss became NaN at step N; lower the learning rate` | Divergence. Lower `--lr` or the batch. |
| `the text has N bytes but needs more than the context of M` | `--data` is too short for `--context`; give more text or a smaller context. |
| `rope scaling type 'yarn' is not supported` (or similar) | The GGUF uses a feature that would be silently wrong if ignored; see [GGUF.md](GGUF.md). |
| A GPU run crashes the driver with several runs in parallel | Concurrent device opens can fault some drivers. Run one GPU job at a time (the tests use `--test-threads 1`). |
| `cannot listen on HOST:PORT` | The port is taken; use `--port 0` for any free port. |
| `refusing to listen on ...: it is not a loopback address` | Exposing the server needs a token: `--token T` (or `FERRUM_TOKEN`), or `--allow-unauthenticated` if you accept the risk. |

## 10. What is not here yet

Server request cancel on disconnect, tied heads, dropout and character-level models for GPT-2 training, `tokenize`/`prepare` and token files, K-quant and other block formats, GPU-side scoring (it runs on the
CPU), automatic VRAM discovery for the budget, Llama tied-head training, QAT for Llama, micro-batching,
Windows and macOS verification. Cross-loading is checked against ferrum-cpu files kept in `fixtures/bpe/` (see docs/PARITY.md).
