# Interoperability: GGUF, llama.cpp, Ollama, LM Studio

What was checked, against which versions, and what was not. Every statement below is backed by a test
in `ferrum_gpu_cli/tests/interop/` (run with `cargo test -p ferrum_gpu_cli --test interop`).

## What runs where

| | Reads our files | We read its files | Registered / imported by `ferrum-gpu` |
|---|---|---|---|
| **GGUF** (the format) | yes: v3, 32-byte alignment, every tensor offset ordered and inside the file | yes: v2/v3, F32 F16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_1 and Q2_K to Q6_K | n/a |
| **llama.cpp** | yes: same token ids, same greedy text, for f32, Q8_0 and Q4_0 files | yes: files from `llama-quantize` as Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K_M Q6_K | n/a |
| **Ollama** | yes: Ollama's raw greedy text equals ours (f32 and Q8_0) | not tested (Ollama stores models as blobs; it is llama.cpp inside) | `export-ollama` registers it with a real server |
| **LM Studio** | via llama.cpp (its engine): the bytes `lms` receives load and give our text | not tested | `export-lmstudio` runs the right `lms` commands (against a fake `lms`) |

Only `gpt2`-architecture files are exported: that is what `train` writes, with a BPE tokenizer embedded.
Llama/Qwen-style files written by `train --arch llama` carry no tokenizer and are for ferrum-gpu itself;
`export-ollama` and `export-lmstudio` refuse them by name.

## Versions actually run

Recorded 2026-11-14 on Zorin OS 18.1 (Linux x86-64), CPU only.

- llama.cpp **b11521** (`llama-completion`, `llama-tokenize`, `llama-quantize`), CPU backend.
- Ollama **0.40.2**, `ollama serve` on a private port with a private `HOME` and model directory,
  `CUDA_VISIBLE_DEVICES=-1`, `OLLAMA_LLM_LIBRARY=cpu`.
- LM Studio's `lms` CLI (commit `dcefd5a`) is installed here but **was not run**: `lms import` registers
  models under `~/.lmstudio`, outside this repository. The tests use a fake `lms` (a shell script that
  records its arguments and the file it was given) and check the command line and the bytes.
- Not run at all: llama.cpp on a GPU backend, Ollama on a GPU, LM Studio's application, macOS and
  Windows builds of any of them.

## How the tests are built

- **Minimal data.** One sentence repeated 60 times trains a model with a 269-token vocabulary, 2
  layers and width 32 (tens of thousands of parameters) in a few seconds; it has learned the sentence
  so far past rounding noise that greedy text is the same
  at f32, Q8_0 and Q4_0, which is what makes exact comparison against another implementation fair.
- **Skip, never fail, without the tool.** Each test that needs llama.cpp or Ollama prints `SKIP: ...`
  and returns when it is missing; `FERRUM_LLAMACPP_DIR` and `FERRUM_OLLAMA` point at other installs.
  The structural GGUF tests and the LM Studio command-line tests need no tool.
- **Servers are contained.** The Ollama server runs in its own process group with a private `HOME`,
  so it never touches `~/.ollama`; it is ended with `kill -s TERM -- -<pgid>` (CLAUDE.md rule 16) and
  a lifetime limit from `timeout`.

## Findings

- llama.cpp accepted our GPT-2 files with no change: the embedded BPE vocabulary, the tensor names and
  the ggml dimension order all matched on the first try.
- `export-ollama` produced a Modelfile Ollama accepted and described as family `gpt2` with our
  `num_ctx` and stop token.
- A client that pipelined extra bytes after a request made the GUI/server close the socket with
  unread data, which resets the connection and can destroy the response; connections now half-close
  and drain briefly first (regression test in `ferrum_gpu_gui/src/http.rs`).
