# The `ferrum-gpu` command line and server (Phase 6, 2026-11-11)

Everything below is real output from the binaries built from this commit (debug profile) on the dev box: Intel Iris Xe + NVIDIA GeForce MX330 + llvmpipe, Vulkan, `wgpu` 30.0.1. The NVIDIA/Mesa driver stack prints `libEGL warning: ...` lines to **stderr** on every run; they are left out here. Tokens are bytes (vocabulary 256); a BPE tokenizer is not part of Phase 6. No timings are claimed in this page.

Build: `cargo build --release -p ferrum_gpu_cli -p ferrum_gpu_server` gives `target/release/ferrum-gpu` and `ferrum-gpu-server`.

## Commands

`train`, `finetune-gguf`, `generate`, `run-gguf`, `eval`, `info`; `ferrum-gpu help` prints every flag. Unknown flags are errors. Device flags: `--device cpu|gpu|auto` (default `auto`), `--gpu-adapter NAME`, `--gpu-mem-mb N`, `--threads N`.

### What `--device auto` does

- **Training** goes to a hardware GPU when one exists (Phase 4: 1.3x to 5.5x faster than the 8-thread CPU baseline on the two adapters tested).
- **Decoding, generation and serving** stay on the CPU (Phase 3: no GPU speed-up up to 34M parameters on those adapters). Larger models were not measured; `--device gpu` overrides.
- With no hardware adapter (none, or only software like llvmpipe) `auto` uses the CPU and prints one `note:` line.

```
$ ferrum-gpu info
device: CPU (8 threads) (cpu)
note: auto: using the CPU for decoding (measured faster than the GPU up to 34M parameters; --device gpu overrides)
adapter: Intel(R) Iris(R) Xe Graphics (TGL GT2) (Integrated, Vulkan)
adapter: NVIDIA GeForce MX330 (Discrete, Vulkan)
adapter: llvmpipe (LLVM 20.1.2, 256 bits) (Software, Vulkan)
adapter: Mesa Intel(R) Iris(R) Xe Graphics (TGL GT2) (Integrated, Gl)
```

### Train, evaluate, generate

The corpus is `the cat sat on the mat. ` repeated 60 times. `auto` picked the MX330 for training:

```
$ ferrum-gpu train --preset tiny --data data.txt --out m.gguf --steps 150 --batch 4 --context 32 --lr 0.01
device: NVIDIA GeForce MX330 (gpu)
model: vocab 256 ctx 64 dim 32 heads 4/2 layers 2 hidden 64
preflight: GPU buffers 1.8 MiB in 172 allocations (budget unknown)
steps: 150 loss 6.2224 -> 0.0408
wrote m.gguf

$ ferrum-gpu eval --model m.gguf --data data.txt
tokens: 1417 loss: 0.0592 perplexity: 1.061

$ ferrum-gpu generate --model m.gguf --prompt "the c" --max-new 24
the cat sat on the mat. the c
```

`generate` prints the prompt followed by the new tokens. `eval` runs on the CPU oracle over non-overlapping windows of the context, and measures the same corpus it trained on (no held-out split). Fine-tuning a GGUF file is `finetune-gguf --model m.gguf --data ... --out g.gguf` with the same training flags; `--quant q8_0|q4_0` quantises the file that is written.

### Memory pre-flight

On a GPU run the engine's own buffer plans (weights, gradients, both Adam moments, activations for the step's batch and context) are checked against the adapter's binding limit and `--gpu-mem-mb` before anything is uploaded. Without `--gpu-mem-mb` the budget is unknown and only the binding limit applies (no VRAM discovery is wired in).

```
$ ferrum-gpu train --device gpu --preset tiny --data data.txt --out g.gguf --steps 2 --gpu-mem-mb 0
device: NVIDIA GeForce MX330 (gpu)
model: vocab 256 ctx 64 dim 32 heads 4/2 layers 2 hidden 64
refusing to start on NVIDIA GeForce MX330: needs 1928832 bytes of GPU memory, budget is 0 (needs 1.8 MiB in total); use a smaller preset, batch or context, raise --gpu-mem-mb, or run with --device cpu
exit 2
```

With `--gpu-mem-mb 1024` the same command trains (`preflight: GPU buffers 1.8 MiB in 172 allocations (budget 1024.0 MiB)`) and ends at the same loss as the run above, `6.2224 -> 0.0408`: the GPU trainer is deterministic.

## The server

`ferrum-gpu-server --model m.gguf [--port N] [--device ...] [--gpu-mem-mb N]` listens on `127.0.0.1` only (requests with a foreign `Host` or `Origin` get 403) and serialises model calls. Routes: `GET /health`, `GET /v1/info`, `POST /v1/generate` (`prompt`, optional `max_new` 1..=1024, `temperature`, `top_k`, `seed`), `POST /v1/loglikelihood` (`context`, `continuation`: sum of the continuation tokens' log-probabilities, whether each was the argmax). Prompts are limited to 4096 bytes. Scoring runs on the CPU oracle.

```
$ ferrum-gpu-server --model m.gguf --device cpu --port 8787
device: CPU (8 threads) (cpu)
ferrum-gpu server: http://127.0.0.1:8787/

$ curl -s http://127.0.0.1:8787/health
{"status":"ok"}
$ curl -s http://127.0.0.1:8787/v1/info
{"device":"CPU (8 threads)","kind":"cpu","params":34976,"vocab":256,"ctx":64}
$ curl -s -d '{"prompt":"the c","max_new":16}' http://127.0.0.1:8787/v1/generate
{"text":"at sat on the ma","tokens":16}
$ curl -s -d '{"context":"the cat sat on the","continuation":" mat"}' http://127.0.0.1:8787/v1/loglikelihood
{"loglikelihood":-0.002561522662290372,"is_greedy":true,"tokens":4}
```

## Tests

`cargo test -p ferrum_gpu_cli -p ferrum_gpu_server -- --test-threads 1`: unit tests (flag parser, every command on the CPU backend, the `auto` rule over fake adapter tables, the pre-flight with a fake GPU backend, every route with a stub decoder, real sockets) plus end-to-end tests that run the real binaries: train, eval, generate, fine-tune and refusals on the CPU; a GPU train and a pre-flight refusal that **skip with a printed reason** when no hardware adapter is reported; the server binary answering all four routes.

## Not done

BPE tokenizer; a real ferrum-cpu GGUF fixture; GPU-side scoring; VRAM discovery for the budget; Windows and macOS runs.
