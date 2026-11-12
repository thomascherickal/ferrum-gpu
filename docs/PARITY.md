# Parity ledger: ferrum-gpu versus ferrum-cpu

Written for M0 of [MATCHER-PLAN.md](../MATCHER-PLAN.md). Audited against **ferrum-cpu commit `56a9d3ada59df2a7f3f4e128d04a96b59211581c`** (read-only, 2026-11-11). ferrum-cpu is a reference for behaviour only; nothing here links to it (see the independence rule in MATCHER-PLAN.md section 0).

**How deep the audit went.** Read in source: the flag tables (`slm_cli/src/args.rs`), presets (`slm_cli/src/preset.rs`), GGUF quantization constants and dequantizers (`ferrum_core/src/gguf/{format,dequant}.rs`), the accepted architectures (`gguf/llama.rs`), the server routes, limits and flags (`slm_server/src/{http,limits,cli}.rs`). Read only as documentation (README, STATUS, `--help` text): behaviour of `prepare`, `tokenize`, checkpoints, the chat template, export commands, LightEval. Those rows say "doc" and are re-read in their own phase before work starts.

Status words: **have** (works here), **partial**, **gap**, **different** (deliberately not the same, reason given).

## Findings that change the plan

1. **(Closed in M4, 2026-11-11.) `train` made a different architecture.** ferrum-cpu `train` writes a `gpt2` GGUF (learned positions, LayerNorm, GELU, byte-level BPE) that also runs in llama.cpp. `ferrum-gpu train` builds a Llama-style model (RMSNorm, RoPE, GQA, SwiGLU; heads shown as `4/2`) and writes a `llama` GGUF. The engine has a GPT-2 trainer (`ferrum_wgpu::training::gpt2`) but the CLI does not expose it. Parity needs `train --arch gpt2|llama` (M4), with `gpt2` the default to match ferrum-cpu.
2. **Presets differ.** See the table below. They are different models under the same names.
3. **Quantization is a bigger gap than listed.** ferrum-cpu reads 11 block formats; we read 3 plus F32/F16.
4. **(Closed in M7.) The server has one more route** than the plan said (`/v1/loglikelihood_rolling`) and a fixed in-flight cap of 32.

## Presets

| name | ferrum-cpu (gpt2, BPE) | ferrum-gpu (llama-style, byte) |
|---|---|---|
| tiny | context 64, embed 64, heads 4, blocks 2, hidden 256, vocab 512, lr 1e-2 | context 64, dim 32, heads 4/2, layers 2, hidden 64, vocab 256 |
| small | context 128, embed 128, heads 4, blocks 4, hidden 512, vocab 2048, lr 3e-3 | context 128, dim 64, heads 4/2, layers 3, hidden 176, vocab 256 |
| medium | context 256, embed 256, heads 8, blocks 6, hidden 1024, vocab 4096, lr 1e-3 | context 128, dim 128, heads 8/2, layers 4, hidden 352, vocab 256 |

Decision: when `--arch gpt2` is selected the ferrum-cpu preset sizes are used (M4); the current llama-style presets stay under `--arch llama`. Preset `lr` values are applied as defaults only if the user gives none.

## Commands and flags

ferrum-cpu flags are from its flag tables. "Phase" is where the gap closes.

| Feature | ferrum-cpu | ferrum-gpu | Status | Phase |
|---|---|---|---|---|
| `train`: size flags `--context --embed --heads --blocks --hidden --vocab` | yes | **have (M4)**, same defaults and presets for `--arch gpt2` | have | |
| `train`: `--epochs`, `--tokens` budget, `--stride` | yes | **have (M4)**, plus our `--steps` | have | |
| `train`: `--lr --batch --seed --threads` | yes | yes | have | |
| `train`: `--weight_decay --cosine --warmup --clip --no-qat` | yes | **have (M4)** | have | |
| `train`: `--tie` | yes | **have (CPU and GPU engines)**: one shared matrix (`head = tokᵀ`), the head's gradient folded into the embedding's before AdamW, the head re-derived after every update; QAT snaps the embedding and derives the head from it; the file stores the embedding once. The GPU engine does it with one transpose kernel (`opt/transpose`); its loss curve and weights follow the CPU's within 1e-4 with and without QAT and with micro-batching (checked on llvmpipe in the tests, on the NVIDIA and Intel adapters in the gate) | have | |
| `train`: `--dropout` | yes | **have (CPU and GPU engines)**: FFN-hidden dropout after GELU, inverted scaling, rate clamped to 0.95, as ferrum-cpu. The mask is a counter hash of (seed, layer, unit) on 32-bit words, so a seeded run is reproducible, a resumed run continues its masks, and the GPU kernel (`fwd/dropout`) drops exactly the units the CPU drops (tested bit for bit); micro-batches number their units from the start of the batch. The gradient is checked against finite differences, and GPU runs follow CPU runs within 1e-4 (plain, QAT, tied, 0.9, micro-batched). Composes with `--tie`. The masks are not ferrum-cpu's (different RNG), so loss curves agree in behaviour, not digit for digit | have | |
| `train`: `--quant int8\|int4\|f32` | yes | **have (M4)**: both spellings accepted | have | |
| `train`: `--tokenizer` (BPE file, trained and saved if missing) | yes | **have (M4)**: trained and saved if missing, loaded if present; a token file must be made with the same tokenizer (SHA-256 checked) | have | |
| `train`: `--checkpoint --checkpoint-every --resume` | yes | **have (M4)**; our checkpoint is a different GGUF kind, not interchangeable with ferrum-cpu's `.fckp` | different | |
| `train`: `--val --patience` | yes | **have (M4)** (validation after each epoch, best epoch kept) | have | |
| `train`: `--max-seconds`, stop file `<model>.stop` | yes | **have (M4)** (the stop file is `<out>.stop` with the extension replaced) | have | |
| `train`: `--metrics-csv`, `--verbose`, `--sample` | yes | **have (M4)**; `-v` short form not accepted; `--debug` and `--force` accepted and ignored | have | |
| `train`: `--preset` | yes | **have (M4)**: ferrum-cpu's sizes for `--arch gpt2`; the old sizes stay under `--arch llama` | have | |
| `train` on a token file (`tokens.gguf`, streamed) | yes (doc) | **have (M2)**: windows read from disk; the tokenizer file's SHA-256 is checked | have | |
| `run` (train then generate in one command) | yes | **have (2026-11-12)**: `run --data F --model M.gguf --prompt TEXT [train and generate flags]`, flag-style; trains M.gguf if it does not exist, loads it if it does | have | |
| `generate`: `--chars --temp --gen-seed --top-k --top-p --repetition-penalty` | yes | **have (M4)** for GPT-2 files (ferrum-cpu's spellings are aliases of ours) | have | |
| `generate`: `--stream`, `--stop` (repeatable), `--max-tokens` | yes | **have (M4)** for GPT-2 files | have | |
| `run-gguf`: `--quant --max --temp --gen-seed --top-k --top-p --rep` | yes | **have (M5)**: same sampling flags as `generate`, for GPT-2 files and for imported Llama/Qwen files with their own tokenizer | have | |
| `run-gguf`: `--stop`, `--ids`, `--raw`, chat template, `--force`, `--resume` | yes | **have (M5)**: `--stop`, `--ids`, `--raw` and the chat template (ChatML, Llama-3, `[INST]`, Zephyr) for imported files; the reply ends at the end-of-turn token. `--resume` (overlaying a checkpoint on an imported model) is deliberately different: ferrum-cpu fine-tunes into a separate `.flck` checkpoint that run-gguf overlays, while `finetune-gguf` here writes a complete GGUF, so there is nothing to overlay; `--force` is accepted and does nothing | partial | |
| `finetune-gguf` on a GPT-2 file (ours; ferrum-cpu's is for Llama/Qwen) | no | **have (M4)**, keeps the file's tokenizer | different | |
| `finetune-gguf` options (`--epochs --lr --batch --seq --weight_decay --dropout --clip --warmup --qat --seed --threads --resume --sample --max-seconds`) | yes | exists, fewer options; ferrum-cpu writes an `.flck` checkpoint, we write a GGUF | partial | M4 |
| `export-gguf` (`--quant`) | yes | **have (M3)**: f32/int8/int4 out | have | |
| `export-ollama` (`--bin`, `--modelfile-only`), `export-lmstudio` (`--repo`, `--bin`) | yes | **have (M6)**, flag-style (`--model`, `--name`); same Modelfile and `lms` commands; tested with fake tools only (the real tools are installed but registering writes outside the repo, so it was not run) | partial | |
| `eval` (`--json`) | yes | **have (M4)**, plus `--ctx` | have | |
| `eval-gguf` (`--ctx --quant --resume --force --json`) | yes | **have (M5)**: `eval-gguf` is an alias of `eval`, which takes `--ctx --quant --json` and scores GPT-2 and imported files with their own tokenizers. `--resume` is deliberately different for the same reason as in `run-gguf` (our fine-tuned model is a complete GGUF) | have | |
| `prepare` (`--input --out --doc-sep --lossy --nfkc --lowercase --strip-gutenberg --min-chars --max-chars --dedup --val-frac --test-frac --seed`) | yes | splits and hashes equal ferrum-cpu on `fixtures/prepare/`; manifest differs only in paths, args and version | done | M2 |
| `tokenize` (`--tokenizer --out --doc-sep`) | yes (doc) | **have (M2)**: `--data FILE\|DIR`; output is byte-identical to ferrum-cpu's on the fixture | have | |
| `info` | prints model info | `info [--model F.gguf]` prints the device and adapters, plus the model's shape when a file is given | have | |
| Unknown flags are errors | yes | yes | have | |

## Formats

| Item | ferrum-cpu | ferrum-gpu | Status | Phase |
|---|---|---|---|---|
| Architectures read | `llama`, `qwen2` (imported), `gpt2` (own) | `gpt2`, `llama`, `qwen2` | have | |
| F32, F16 | yes | yes | have | |
| Q8_0, Q4_0 | yes | yes | have | |
| Q4_1, Q5_0, Q5_1, Q8_1 | yes | **have (M3, 2026-11-11)**: read, dequantized to f32 on load | have | |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | **have (M3)**: read, dequantized to f32 on load; no GPU kernel reads them in place and nothing is written in these formats | partial | |
| BPE tokenizer (`ferrum_tok`: trainer, encoder, GPT-2 pre-tokenizer, specials), GGUF tokenizer file and model metadata (`ferrum_model::tokenizer_gguf`) | yes | **have (M1, 2026-11-11)**: reads and writes ferrum-cpu's tokenizer file and model metadata; encodes the fixture corpus to the same 942 ids; our GPT-2 reader loads a real ferrum-cpu model file | have | |
| Character-level model (`--vocab 0`) | distinct-character vocabulary, ferrum-cpu-only file | one token per byte (256 ids), ordinary tokenizer file and `gpt2` GGUF | differs (by design) | M4 |
| Refusals | refuses biases on any projection except q/k/v, and any RoPE scaling type other than `linear` or `none` (with factor 1.0) | refuses yarn, longrope, attention factor other than 1, and biases outside attention q/k/v | have (same intent; ours is at least as strict; exact messages not compared) | |

## Server

| Feature | ferrum-cpu | ferrum-gpu | Status | Phase |
|---|---|---|---|---|
| `GET /health`, `GET /v1/info`, `POST /v1/generate`, `POST /v1/loglikelihood` | yes | yes | have | |
| `POST /v1/loglikelihood_rolling` | yes | **have (M7)**, with ferrum-cpu's window rule | have | |
| Flags `--model --host --port` | yes | **have (M7)** (default port 8080) | have | |
| `--max-batch --max-chars --max-body --max-tokens --repetition-penalty` | yes | **have (M7)**, same defaults (64, 65536, 8 MiB, 4096, 1.0) | have | |
| `--quant` | yes | not applicable (load as stored) | different | M7 |
| Bearer token (`--token`, `FERRUM_TOKEN`), non-loopback needs a token, `--allow-unauthenticated`, `--force` | yes | **have (M7)**; `--force` accepted and ignored | have | |
| In-flight cap (32 concurrent POSTs), stop-sequence limits | yes | **have (M7)** (503 over the cap; 16 stops of 1024 bytes) | have | |
| Imported-model routes | yes | one decoder trait; imported Llama/Qwen served | have | |
| Framework | axum + tokio | std only | different, on purpose (keeps dependencies at zero) | |

## Other

| Feature | ferrum-cpu | ferrum-gpu | Status |
|---|---|---|---|
| LightEval module (Python) | yes | ferrum-cpu's adapter code was driven against our server 2026-11-12 (info, log-likelihood, rolling, generate); it needed `context_len` in `/v1/info`, now added. The full benchmark tasks (they download datasets) were not run | have (adapter), tasks not run |
| DVC pipeline (`dvc.yaml`, `params.yaml`) | yes | files exist at the repository root; DVC itself was never run | partial |
| `scripts/fetch.sh` pinned download | yes | `scripts/fetch.sh` exists (not run against the network) | partial |
| Browser | `slm_wasm` CPU in WASM | WebGPU in the tab | different (Phase 5b) |
| GUI: tabs | 12 (Datasets, Prepare, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, Serve, System, Capable) | 4 (Device, Model, Train, Generate); the CLI does the rest | different, on purpose: the GUI is a window onto training and sampling, not a second front end |
| GUI: Train options | architecture, epochs/token budget, stride, LR/cosine/warmup, validation share, checkpoints | steps, LR, batch, context, **validation share (2026-11-12)**; warmup and clip are fixed | partial |
| GUI: Generate options | temperature, top-k, top-p, repetition penalty, stop strings | temperature, top-k, **stop strings (2026-11-12)**; no top-p or repetition penalty (the byte engine has none) | partial |
| GUI: checkpoints | save/load a model | **have**: Download GGUF (f32, q8_0, q4_0) and Load a GGUF | have |
| GUI: BPE tokenizer | yes | no: byte-level models only, see docs/gui.md | different, on purpose |
| GUI: shell | Tauri desktop app | local web page on 127.0.0.1 | different |
| RAM guard (Linux), `--force` | yes | GPU buffer pre-flight and `--gpu-mem-mb` | different |
| Determinism | bit-identical across thread counts | bit-identical across repeat runs per adapter | different (stated in ARCHITECTURE.md) |
| Device choice | CPU only | `--device cpu\|gpu\|auto` | GPU-only feature |
