# Interoperability: GGUF, llama.cpp, Ollama, LM Studio

What was checked, against which versions, and what was not. Every statement below is backed by a test
in `ferrum_gpu_cli/tests/interop/` (run with `cargo test -p ferrum_gpu_cli --test interop`).

## What runs where

| | Reads our files | We read its files | Registered / imported by `ferrum-gpu` |
|---|---|---|---|
| **GGUF** (the format) | yes: v3, 32-byte alignment, every tensor offset ordered and inside the file | yes: v2/v3, F32 F16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_1 and Q2_K to Q6_K | n/a |
| **llama.cpp** | yes: same token ids, same greedy text, for f32, Q8_0 and Q4_0 files | yes: files from `llama-quantize` as Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K_M Q6_K | n/a |
| **Ollama** | yes: Ollama's raw greedy text equals ours (f32 and Q8_0) | not tested (Ollama stores models as blobs; it is llama.cpp inside) | `export-ollama` registers it with a real server |
| **LM Studio** | via llama.cpp (its engine): the bytes `lms` receives load and give our text | not tested | `export-lmstudio` runs the right `lms` commands (fake `lms` by default; a real `lms import` is opt-in, see below) |

Only `gpt2`-architecture files are exported: that is what `train` writes, with a BPE tokenizer embedded.
Llama/Qwen-style files written by `train --arch llama` carry no tokenizer and are for ferrum-gpu itself;
`export-ollama` and `export-lmstudio` refuse them by name.

## Versions actually run

Recorded 2026-11-14 on Zorin OS 18.1 (Linux x86-64); the default suite is CPU only.

- llama.cpp **b11521** (`llama-completion`, `llama-tokenize`, `llama-quantize`), CPU backend.
- Ollama **0.40.2**, `ollama serve` on a private port with a private `HOME` and model directory,
  `CUDA_VISIBLE_DEVICES=-1`, `OLLAMA_LLM_LIBRARY=cpu`.
- Real GPU and LM Studio runs, 2026-11-14 (opt-in, full step log in `docs/logs/interop-real-run-2026-11-14.log`):
  - llama.cpp b11521 **Vulkan build** (the installed one is CPU only; the matching release zip was put
    in a scratch directory, not installed): `-ngl 99 --device Vulkan0` (Intel Iris Xe) and
    `--device Vulkan1` (NVIDIA MX330), f32, Q8_0 and Q4_0. Log shows `offloaded 3/3 layers to GPU`;
    the text equals ours in every case. A CPU control run shows no layer offloaded.
  - Ollama 0.40.2 with `CUDA_VISIBLE_DEVICES` untouched (log shows `CUDA0` buffers) and with
    `OLLAMA_VULKAN=1` (`Vulkan0` buffers), f32 and Q8_0: `/api/ps` reports `size_vram == size`, text equals ours.
    Ollama drops the integrated Intel GPU by default, so both runs used the MX330.
  - A real `lms import -y -l` (LM Studio's `lms`, commit `dcefd5a`, server off): registered as
    `ferrum-interop-test/tiny-gpt2`, `lms ls --json` went from 10 to 11 models, the file is a
    byte-identical symlink and runs in llama.cpp with our text. The folder was removed afterwards;
    `~/.lmstudio/models` is back to its 15 entries.
- Not run: LM Studio's application or its server, GPU runs of LM Studio itself, macOS and Windows builds
  of any of these.

## Unsloth Studio

Not tested against a real Studio. Studio is a web app with its own Python environment and a llama.cpp
backend that loads local GGUF files from its UI; it has no documented import command that
`ferrum-gpu` could call, so there is no `export-unsloth`. The interoperability is the GGUF file itself:
the llama.cpp tests above (CPU and Vulkan, f32/Q8_0/Q4_0) show that our `gpt2` GGUF files load and give our text
in llama.cpp, which is the engine Studio uses. That is an inference, not a Studio run. Studio was not
installed (a multi-GB Python environment outside this repository).

## Running the real-tool tests (opt-in)

Off by default so the suite never touches `~/.lmstudio` or the GPU through outside tools:

```
FERRUM_INTEROP_GPU=1 FERRUM_LLAMACPP_VULKAN_DIR=<dir with llama-completion + libggml-vulkan.so> \
  cargo test -p ferrum_gpu_cli --test interop -- llamacpp_gpu ollama_gpu --test-threads 1 --nocapture
FERRUM_INTEROP_LMS=1 cargo test -p ferrum_gpu_cli --test interop -- lms --test-threads 1 --nocapture
```

Set `FERRUM_INTEROP_LOG=<file>` to append every step (timestamped) to a file. Time limits: ordinary
tests have none; each interop test may take 10 minutes (`scripts/test-time-limit.sh`, `INTEROP_LIMIT`).

## Findings

- **llama.cpp b11521 is not deterministic at its default thread count.** The same `llama-completion`
  command on the same file, eight times: seven runs gave the full text and one gave text cut short
  with exit status 0. With `-t 1` or `-t 2`, eight of eight identical (checked on both). The tests
  run it with `-t 1` (and Ollama with `num_thread: 1`), so a mismatch means a real difference.

- llama.cpp accepted our GPT-2 files with no change: the embedded BPE vocabulary, the tensor names and
  the ggml dimension order all matched on the first try.
- `export-ollama` produced a Modelfile Ollama accepted and described as family `gpt2` with our
  `num_ctx` and stop token.
- A client that pipelined extra bytes after a request made the GUI/server close the socket with
  unread data, which resets the connection and can destroy the response; connections now half-close
  and drain briefly first (regression test in `ferrum_gpu_gui/src/http.rs`).
