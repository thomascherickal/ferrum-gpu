# ferrum-gpu

Train, fine-tune, score and run **small causal Transformers on the GPU**, in safe Rust with
[`wgpu`](https://wgpu.rs) and WGSL. Models come in and go out as **GGUF** files. The engine targets any GPU
`wgpu` can reach (Vulkan, Metal, DX12), including Apple Silicon, but it has so far been run only on the
hardware listed under [Status](#status).

ferrum-gpu mirrors the behaviour of the CPU project ferrum-cpu **without depending on it**: the two
meet only at the GGUF file format. Every GPU kernel is checked against a CPU reference written in this
repository (`ferrum_ref`), and every figure in the docs comes from a real run.

## What it does

| | |
|---|---|
| Models | GPT-2 style (learned positions, LayerNorm, GELU) and Llama / Qwen2 style (RMSNorm, RoPE, grouped-query attention, SwiGLU, optional q/k/v biases, tied or untied head) |
| Inference | KV-cached single-token decode and full-sequence scoring, with f32, Q8_0 or Q4_0 weights kept packed on the device |
| Training | Forward, backward, global-norm clipping and AdamW entirely on the device; optional Q8_0 quantisation-aware training; checkpoints as GGUF; bit-identical across repeat runs on the same adapter, driver and `wgpu` version |
| Formats | GGUF v3 in and out (`gpt2`, `llama`, `qwen2`; f32, f16 read, Q8_0, Q4_0) |
| Front ends | `ferrum-gpu` command line, `ferrum-gpu-server` JSON API, a local web GUI, and in-browser inference on WebGPU |

## Quick start

Requirements: Rust 1.87 or newer, and a GPU driver with Vulkan, Metal or DX12 if you want the GPU path
(everything also runs on the CPU, and the test suite passes on a machine with no GPU).

```bash
cargo build --release -p ferrum_gpu_cli -p ferrum_gpu_server -p ferrum_gpu_gui

# train a tiny GPT-2 model with its own BPE tokenizer on a text file, then use it.
# The file is a GPT-2 GGUF that ferrum-cpu and llama.cpp-style tools can load.
target/release/ferrum-gpu info
target/release/ferrum-gpu train    --preset tiny --data corpus.txt --out m.gguf --epochs 30 --batch 16 --vocab 300
target/release/ferrum-gpu eval     --model m.gguf --data corpus.txt
target/release/ferrum-gpu generate --model m.gguf --prompt "the c" --max-new 24

# a Llama-style byte-level model (served by ferrum-gpu-server until the server learns GPT-2 files), or the GUI
target/release/ferrum-gpu train --arch llama --preset tiny --data corpus.txt --out l.gguf --steps 150 --batch 4 --context 32 --lr 0.01
target/release/ferrum-gpu-server --model l.gguf --port 8787
target/release/ferrum_gpu_gui
```

`--device cpu|gpu|auto` (default `auto`) picks where it runs; see the [user guide](docs/USER-GUIDE.md) for
what `auto` does and why.

### Run every test

`scripts/test-all.sh` runs all tests in all crates (with and without a GPU), prints a test and coverage summary, and writes `LAST-TEST-RESULTS.md` and `LAST-COVERAGE-RESULTS.md`. See [docs/DEVELOPING.md](docs/DEVELOPING.md).

### Run the GUI

```bash
scripts/run-gui.sh               # http://127.0.0.1:8080/, --device gpu
scripts/run-gui.sh --retune      # redo the adapter and thread choice
scripts/run-gui.sh --port 9000   # extra flags go to the GUI
```

The script builds in release mode, then starts the GUI on port 8080 with `--device gpu`. On the first run
it picks the GPU adapter and thread count for this architecture (OS + CPU, for example `Linux-x86_64`) and
saves them in `.gui-prefs` (git-ignored); later runs reuse them. The adapter is the best hardware one by
rank: Discrete over Integrated, Vulkan/Metal/DX12 over Gl, never Software such as llvmpipe. This is a
ranking rule, not a benchmark. Threads are the machine's core count. With no hardware adapter the script
exits with an error. The GUI listens on loopback only and has no authentication.

## Status

Phases 0 to 6 are complete: kernels and parity, GGUF I/O, GPT-2 and Llama/Qwen inference, training and
fine-tuning, the GUI and browser WebGPU runtime, and the CLI and server. Phase 7 (a verification matrix
over more devices, then making `auto` the default everywhere) has not started. The plan is in
[PLAN.md](PLAN.md); per-phase notes and outcomes are in [docs/plans/](docs/plans/).

Hardware actually run: NVIDIA GeForce MX330 (2 GB, Vulkan), Intel Iris Xe (Vulkan), llvmpipe (software
Vulkan), on Linux. **Not run:** Apple Silicon, AMD, desktop NVIDIA, Windows, macOS, Firefox, Safari.

What the measurements say, in one paragraph (details and caveats in the linked reports): batched work
wins on the GPU. A full training step is 1.3x to 5.5x faster than an 8-thread CPU step on the two real
adapters tested ([train-speed](docs/train-speed.md)); one-token-at-a-time decode is **not** faster than a
single CPU thread up to 34M parameters ([decode-crossover](docs/decode-crossover.md),
[llama-speed](docs/llama-speed.md)). `--device auto` follows those results.

## Repository map

| Crate | Role |
|---|---|
| [`ferrum_gguf`](ferrum_gguf) | GGUF v3 reader and writer, f16 and block-quant codecs. std only, hostile-input safe |
| [`ferrum_ref`](ferrum_ref) | CPU reference: kernels, GPT-2 and Llama models, gradients, AdamW, QAT, the training loop. The parity oracle |
| [`ferrum_model`](ferrum_model) | GGUF mappings for `gpt2` / `llama` / `qwen2`, training checkpoints, token sampling |
| [`ferrum_wgpu`](ferrum_wgpu) | The GPU engine: device runtime, WGSL kernels, resident models, trainers |
| [`ferrum_gpu_cli`](ferrum_gpu_cli) | The `ferrum-gpu` command line, the `auto` device rule, the memory pre-flight |
| [`ferrum_gpu_server`](ferrum_gpu_server) | The `ferrum-gpu-server` JSON API |
| [`ferrum_gpu_gui`](ferrum_gpu_gui) | Local web GUI, std-only HTTP server, in-browser WebGPU runtime |
| [`tests`](tests) | Cross-crate guards (independence from ferrum-cpu, the context-window scripts) |

## Documentation

| Document | For |
|---|---|
| [docs/USER-GUIDE.md](docs/USER-GUIDE.md) | Using the CLI, server and GUI: devices, training, quantisation, memory, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How it is built: layers, the device runtime, kernels, determinism, training, memory planning |
| [docs/GGUF.md](docs/GGUF.md) | Exactly which GGUF files load and what is refused |
| [docs/DEVELOPING.md](docs/DEVELOPING.md) | Building, testing, coverage, benchmarks, adding a kernel or a command, project rules |
| [docs/cli.md](docs/cli.md) | CLI and server walkthrough with real output |
| [docs/gui.md](docs/gui.md), [docs/browser-webgpu.md](docs/browser-webgpu.md) | The GUI and the in-browser runtime |
| [docs/spike-matmul.md](docs/spike-matmul.md), [docs/train-speed.md](docs/train-speed.md), [docs/decode-crossover.md](docs/decode-crossover.md), [docs/llama-speed.md](docs/llama-speed.md) | Measurement reports |
| [docs/COVERAGE.md](docs/COVERAGE.md) | Test coverage numbers |
| [PLAN.md](PLAN.md), [docs/plans/](docs/plans/) | The phase plan and what each phase delivered |
| [CLAUDE.md](CLAUDE.md) | The binding project rules |

## Independence from ferrum-cpu

No dependency (path, git or otherwise) on ferrum-cpu, no file read outside this repository, and
ferrum-cpu is never modified. Code that originated there is copied in with a provenance header naming the
source file and commit. A test (`tests/tests/test_independence.rs`) enforces this.

## License

MIT, see [`LICENSE`](LICENSE) (also `license` in the workspace `Cargo.toml`).
