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

## GPT-2 models with BPE (M4, 2026-11-11)

`train` now defaults to `--arch gpt2`, the architecture ferrum-cpu trains. The output below is real, from the debug-free release build on the dev box (Intel Iris Xe + NVIDIA GeForce MX330 + llvmpipe, Vulkan, `wgpu` 30.0.1, ferrum-cpu `56a9d3a` for the cross-load). The earlier sections of this page used the Llama-style path, which is now `--arch llama`.

The corpus is `the cat sat on the mat. ` repeated 80 times. On the CPU:

```
$ ferrum-gpu train --data c.txt --out m.gguf --preset tiny --vocab 300 --epochs 30 --device cpu --seed 1 --tokenizer tok.gguf --metrics-csv m.csv --verbose
tokenizer: trained 269 tokens
wrote tok.gguf
device: CPU (8 threads) (cpu)
model: vocab 269 ctx 64 dim 64 heads 4 layers 2 hidden 256
corpus: 561 tokens, 8 windows, 30 steps of 16
step 10/30 loss 1.7545 grad_norm 0.300 tokens/s 22801 elapsed 0.4s eta 1s
step 20/30 loss 1.7407 grad_norm 0.149 tokens/s 22417 elapsed 0.9s eta 0s
step 30/30 loss 1.7336 grad_norm 0.113 tokens/s 22314 elapsed 1.4s eta 0s
steps: 30 loss 5.5769 -> 1.7336
wrote m.gguf
```

`--vocab 300` asked for 300 tokens; the text only supports 269 merges, so the tokenizer stopped there. The same command with `--device gpu` (the MX330) printed `preflight: GPU buffers 23.5 MiB in 202 allocations (budget unknown)` and the same loss curve to four decimals, `5.5769 -> 1.7336`.

```
$ ferrum-gpu generate --model m.gguf --prompt "the cat" --max-new 20 --device cpu
the cat the the the the the the the the the the the the the the the the the the the the

$ ferrum-gpu eval --model m.gguf --data c.txt --json e.json
tokens: 552 loss: 1.7604 perplexity: 5.815
```

### Cross-load with ferrum-cpu

The file this run wrote loads in ferrum-cpu's own binary, built from commit `56a9d3a` in a scratch directory (ferrum-cpu untouched), and it continues the prompt the same way:

```
$ train_transformer info m.gguf
Model     : m.gguf  (173088 bytes)
Format    : GGUF v3 (gpt2, int8)
Params    : 138624
Name      : ferrum-gpu Transformer
Task      : TransformerSLM
Input dim : 64
Output dim: 269
Tokenizer : byte-level BPE (269 tokens, 12 merges)

$ train_transformer generate m.gguf "the cat" --chars 20 --temp 0
the cat the the the the the
```

(`--chars` counts characters in ferrum-cpu, tokens here, so ferrum-cpu's text is shorter; both are the same greedy continuation.) This direction is a manual check and is not an automated test. The other direction is automated: `fixtures/bpe/` holds a tokenizer file, a model and a token file written by ferrum-cpu, and our tests load them (`ferrum_model/src/tokenizer_gguf/tests.rs`).

### Options that were run

Validation with early stopping (`--val 0.2 --patience 3 --cosine --warmup 2 --clip 1 --weight_decay 0.01 --quant int4 --checkpoint ck.gguf --sample`) printed `epoch N val loss ...` lines with `(best)` marks and ended with `steps: 20 loss 5.5896 -> 0.7376` and a greedy `sample:`. A `--resume` with a changed validation split was refused: `cannot resume: the run shape changed (...)`. A `<out>.stop` file ended a run after one step (`stopped early: ... found; stopped after step 1`). `--tie` and `--dropout 0.1` are refused by name.
