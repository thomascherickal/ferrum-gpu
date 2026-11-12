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

Changed in M7 (2026-11-11): the routes, request shapes and limits now follow ferrum-cpu's `ferrum-serve`, GPT-2 files are served with their BPE tokenizer, and a bearer token is supported. The earlier single-object shapes (`{"prompt":..,"max_new":..}`) are gone.

`ferrum-gpu-server --model m.gguf [--host 127.0.0.1] [--port 8080] [--token T] [--allow-unauthenticated] [--max-batch 64] [--max-chars 65536] [--max-body 8388608] [--max-tokens 4096] [--repetition-penalty 1.0] [--device ...] [--gpu-mem-mb N]`.

- **Exposure.** On a loopback address (the default) requests with a foreign `Host` or `Origin` get 403. A non-loopback `--host` is refused at startup unless a token is set (`--token` or the `FERRUM_TOKEN` environment variable) or `--allow-unauthenticated` is given. With a token every route except `GET /health` needs `Authorization: Bearer <token>` (401 otherwise); on a non-loopback host the Host check is skipped because the token protects the server.
- **Routes.** `GET /health`; `GET /v1/info`; `POST /v1/generate` `{"requests":[{"prompt","max_tokens","stop":[..],"temperature","top_k","top_p","seed","repetition_penalty"}]}` gives `{"results":[{"text","n_tokens"}]}`; `POST /v1/loglikelihood` `{"requests":[{"context","continuation"}]}` gives `{"results":[{"logprob","n_tokens","greedy"}]}`; `POST /v1/loglikelihood_rolling` `{"texts":[..]}` gives `{"results":[{"logprob","n_tokens"}]}`. Unknown fields are 400; model errors are 422; more than 32 POSTs in flight get 503. Scoring follows ferrum-cpu's rules (trailing context whitespace moves to the continuation; an empty context or a rolling text is scored after the end-of-text token; long texts use half-window sliding) and runs on the CPU oracle. `top_p` and `repetition_penalty` need a GPT-2 file; `chat` is refused (no chat template).
- Request cancel: when a client closes its connection while a POST is running, generation stops within one token (the server polls the connection every 25 ms) and the in-flight slot is freed. A client that half-closes its write side after sending (`shutdown(SHUT_WR)`) counts as gone; browsers, curl, requests and fetch do not do that.
- Not implemented from ferrum-cpu: imported-model `chat`.

```
$ ferrum-gpu-server --model m.gguf --device cpu --port 8787 --token s3cret
device: CPU (8 threads) (cpu)
ferrum-gpu server: http://127.0.0.1:8787/

$ curl -s -i http://127.0.0.1:8787/v1/info | head -1
HTTP/1.1 401 Unauthorized
$ curl -s -H 'Authorization: Bearer s3cret' http://127.0.0.1:8787/v1/info
{"device":"CPU (8 threads)","kind":"cpu","arch":"gpt2","params":138624,"vocab":269,"ctx":64}
$ curl -s -H 'Authorization: Bearer s3cret' -d '{"requests":[{"prompt":"the c","max_tokens":12,"stop":["mat"]}]}' http://127.0.0.1:8787/v1/generate
{"results":[{"text":" the the the the the the the the the the the the","n_tokens":12}]}
$ curl -s -H 'Authorization: Bearer s3cret' -d '{"requests":[{"context":"the cat sat on the","continuation":" mat"}]}' http://127.0.0.1:8787/v1/loglikelihood
{"results":[{"logprob":-1.9499623775482178,"n_tokens":1,"greedy":false}]}
$ curl -s -H 'Authorization: Bearer s3cret' -d '{"texts":["the cat sat on the mat. the cat sat on the mat."]}' http://127.0.0.1:8787/v1/loglikelihood_rolling
{"results":[{"logprob":-40.768139600753784,"n_tokens":14}]}
```

(That model is the 30-step GPT-2 model from the section above; it is barely trained, hence the weak numbers. The 401 reason phrase printed here as `Unknown` before this commit and was fixed in it.)

## Tests

`cargo test -p ferrum_gpu_cli -p ferrum_gpu_server -- --test-threads 1`: unit tests (flag parser, every command on the CPU backend, the `auto` rule over fake adapter tables, the pre-flight with a fake GPU backend, every route with a stub decoder, real sockets) plus end-to-end tests that run the real binaries: train, eval, generate, fine-tune and refusals on the CPU; a GPU train and a pre-flight refusal that **skip with a printed reason** when no hardware adapter is reported; the server binary answering all five routes for a Llama-style and a GPT-2 file, the token gate through `--token` and `FERRUM_TOKEN`, and the refusal to expose a server without a token.

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

## Token files (M2, 2026-11-11)

`tokenize` encodes a text file (or the files directly inside a directory, in path order, with an end-of-text id between files) into a token file, the same `ferrum-tokens` GGUF layout ferrum-cpu writes. `train --data FILE.tok.gguf --tokenizer TOK.gguf` then reads windows from disk instead of holding the corpus; the tokenizer file must be the one the ids were made with (its SHA-256 is recorded in the token file and checked).

Real output, using the corpus and tokenizer file in `fixtures/bpe/` (made by ferrum-cpu):

```
$ ferrum-gpu tokenize --data corpus.txt --tokenizer tok.gguf --out mine.tok.gguf
tokenized corpus.txt -> mine.tok.gguf: 942 tokens, 31 documents, vocab 320 (2010 bytes in, 2332 bytes out)
$ cmp mine.tok.gguf fixtures/bpe/corpus.tok.gguf && echo IDENTICAL
IDENTICAL
$ ferrum-gpu train --data mine.tok.gguf --tokenizer tok.gguf --out m.gguf --preset tiny --epochs 5 --batch 8 --device cpu
tokenizer: tok.gguf (320 tokens)
device: CPU (8 threads) (cpu)
model: vocab 320 ctx 64 dim 64 heads 4 layers 2 hidden 256
corpus: 942 tokens, 14 windows, 10 steps of 8
steps: 10 loss 5.7723 -> 4.7297
wrote m.gguf
```

Our file is byte-identical to the one ferrum-cpu's `tokenize` wrote (`ferrum_model::token_file` tests compare the bytes, and our SHA-256 reproduces the tokenizer digest ferrum-cpu recorded). Training from the token file gives the same model, byte for byte, as training from the text with the same tokenizer (a test). Memory: the file is read window by window; the writer keeps O(1) memory.

## prepare (M2, 2026-11-11)

`ferrum-gpu prepare --input FILE|DIR (repeatable) --out DIR [--nfkc] [--lowercase] [--strip-gutenberg] [--dedup] [--min-chars N] [--max-chars N] [--val-frac X] [--test-frac X] [--seed N] [--doc-sep S] [--lossy]` cleans each document (invisible characters, smart punctuation, blank lines, spaces), drops empty, short, long and duplicate documents, splits by a seeded content hash, and writes `train.txt`, `val.txt`, `test.txt` and `manifest.json` (SHA-256 of every input and output). Output is written to `*.part` files and renamed at the end, so a failed run leaves the old files alone.

On the 7-document corpus in `fixtures/prepare/` the three split files are byte-identical to what ferrum-cpu's `prepare` wrote, and the manifest hashes and counts are equal (a test). Not verified: a large corpus run against ferrum-cpu. Not ported: progress callbacks and the stop hook.

`dvc.yaml`, `params.yaml` and `scripts/fetch.sh` give the fetch, prepare, train, eval pipeline. DVC itself was not run (not installed here); the same stages were run by hand on the BPE fixture corpus: prepare, train (loss 5.6535), eval (perplexity 278.05).
