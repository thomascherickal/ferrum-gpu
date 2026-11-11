# Parity ledger: ferrum-gpu versus ferrum-cpu

Written for M0 of [MATCHER-PLAN.md](../MATCHER-PLAN.md). Audited against **ferrum-cpu commit `56a9d3ada59df2a7f3f4e128d04a96b59211581c`** (read-only, 2026-11-11). ferrum-cpu is a reference for behaviour only; nothing here links to it (see the independence rule in MATCHER-PLAN.md section 0).

**How deep the audit went.** Read in source: the flag tables (`slm_cli/src/args.rs`), presets (`slm_cli/src/preset.rs`), GGUF quantization constants and dequantizers (`ferrum_core/src/gguf/{format,dequant}.rs`), the accepted architectures (`gguf/llama.rs`), the server routes, limits and flags (`slm_server/src/{http,limits,cli}.rs`). Read only as documentation (README, STATUS, `--help` text): behaviour of `prepare`, `tokenize`, checkpoints, the chat template, export commands, LightEval. Those rows say "doc" and are re-read in their own phase before work starts.

Status words: **have** (works here), **partial**, **gap**, **different** (deliberately not the same, reason given).

## Findings that change the plan

1. **`train` makes a different architecture.** ferrum-cpu `train` writes a `gpt2` GGUF (learned positions, LayerNorm, GELU, byte-level BPE) that also runs in llama.cpp. `ferrum-gpu train` builds a Llama-style model (RMSNorm, RoPE, GQA, SwiGLU; heads shown as `4/2`) and writes a `llama` GGUF. The engine has a GPT-2 trainer (`ferrum_wgpu::training::gpt2`) but the CLI does not expose it. Parity needs `train --arch gpt2|llama` (M4), with `gpt2` the default to match ferrum-cpu.
2. **Presets differ.** See the table below. They are different models under the same names.
3. **Quantization is a bigger gap than listed.** ferrum-cpu reads 11 block formats; we read 3 plus F32/F16.
4. **The server has one more route** than the plan said (`/v1/loglikelihood_rolling`) and a fixed in-flight cap of 32.

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
| `train`: size flags `--context --embed --heads --blocks --hidden --vocab` | yes | `--context` only; sizes come from `--preset` | gap | M4 |
| `train`: `--epochs`, `--tokens` budget, `--stride` | yes | `--steps` only | gap | M4 |
| `train`: `--lr --batch --seed --threads` | yes | yes | have | |
| `train`: `--weight_decay --dropout --cosine --warmup --clip --tie --no-qat` | yes | engine supports most; CLI exposes none | partial | M4 |
| `train`: `--quant int8\|int4\|f32` | yes | `--quant f32\|q8_0\|q4_0` | partial (names differ) | M4 (accept both spellings) |
| `train`: `--tokenizer` (BPE file, trained and saved if missing) | yes | none (byte tokenizer) | gap | M1, M4 |
| `train`: `--checkpoint --checkpoint-every --resume` | yes | engine has checkpoint and resume; CLI flags missing | partial | M4 |
| `train`: `--val --patience` | yes | none | gap | M4 |
| `train`: `--max-seconds`, stop file `<model>.stop` | yes | none | gap | M4 |
| `train`: `--metrics-csv`, `--verbose`/`-v`, `--debug`, `--sample`, `--force` | yes | none | gap | M4 |
| `train`: `--preset` | yes | yes (different sizes) | different | M4 |
| `train` on a token file (`tokens.gguf`, streamed) | yes (doc) | none | gap | M2 |
| `run` (train then generate in one command) | yes | none | gap | M4 (thin wrapper) |
| `generate`: `--chars --temp --gen-seed --top-k --top-p --repetition-penalty` | yes | sampling in `ferrum_model::sampling`; CLI flags partly exposed | partial | M5 |
| `generate`: `--stream`, `--stop` (repeatable), `--max-tokens` | yes | none | gap | M5 |
| `run-gguf`: `--quant --max --temp --gen-seed --top-k --top-p --rep` | yes | `run-gguf` exists, subset of flags | partial | M5 |
| `run-gguf`: `--stop`, `--ids`, `--raw`, chat template, `--force`, `--resume` | yes | none | gap | M5 |
| `finetune-gguf` options (`--epochs --lr --batch --seq --weight_decay --dropout --clip --warmup --qat --seed --threads --resume --sample --max-seconds`) | yes | exists, fewer options; ferrum-cpu writes an `.flck` checkpoint, we write a GGUF | partial | M4 |
| `export-gguf` (`--quant`) | yes | none | gap | M3 |
| `export-ollama` (`--bin`, `--modelfile-only`), `export-lmstudio` (`--repo`, `--bin`) | yes | none | gap | M6 |
| `eval` (`--json`) | yes | `eval` exists, no `--json` | partial | M5 |
| `eval-gguf` (`--ctx --quant --resume --force --json`) | yes | none (our `eval` runs on any model) | gap | M5 |
| `prepare` (`--input --out --doc-sep --lossy --nfkc --lowercase --strip-gutenberg --min-chars --max-chars --dedup --val-frac --test-frac --seed`) | yes (doc) | none | gap | M2 |
| `tokenize` (`--tokenizer --out --doc-sep`) | yes (doc) | none | gap | M2 |
| `info` | prints model info | prints device and adapters | different | M4 (add model info; keep device lines under `--device`) |
| Unknown flags are errors | yes | yes | have | |

## Formats

| Item | ferrum-cpu | ferrum-gpu | Status | Phase |
|---|---|---|---|---|
| Architectures read | `llama`, `qwen2` (imported), `gpt2` (own) | `gpt2`, `llama`, `qwen2` | have | |
| F32, F16 | yes | yes | have | |
| Q8_0, Q4_0 | yes | yes | have | |
| Q4_1, Q5_0, Q5_1, Q8_1 | yes | no | gap | M3 |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | no | gap | M3 |
| BPE tokenizer in GGUF metadata (write and read) | yes | no | gap | M1 |
| Character-level model (`--vocab 0`) | yes | no | gap | M1 |
| Refusals | refuses biases on any projection except q/k/v, and any RoPE scaling type other than `linear` or `none` (with factor 1.0) | refuses yarn, longrope, attention factor other than 1, and biases outside attention q/k/v | have (same intent; ours is at least as strict; exact messages not compared) | |

## Server

| Feature | ferrum-cpu | ferrum-gpu | Status | Phase |
|---|---|---|---|---|
| `GET /health`, `GET /v1/info`, `POST /v1/generate`, `POST /v1/loglikelihood` | yes | yes | have | |
| `POST /v1/loglikelihood_rolling` | yes | no | gap | M7 |
| Flags `--model --host --port` | yes | `--model --port` (loopback only) | partial | M7 |
| `--max-batch --max-chars --max-body --max-tokens --repetition-penalty` | yes | fixed limits (prompt 4096, new 1024, body 1 MiB) | partial | M7 |
| `--quant` | yes | not applicable (load as stored) | different | M7 |
| Bearer token (`--token`, `FERRUM_TOKEN`), non-loopback needs a token, `--allow-unauthenticated`, `--force` | yes | none (loopback, Host/Origin guard) | gap | M7 |
| In-flight cap (32 concurrent POSTs), stop-sequence limits | yes | none | gap | M7 |
| Imported-model routes | yes | one decoder trait; imported Llama/Qwen served | have | |
| Framework | axum + tokio | std only | different, on purpose (keeps dependencies at zero) | |

## Other

| Feature | ferrum-cpu | ferrum-gpu | Status |
|---|---|---|---|
| LightEval module (Python) | yes | none | gap (M7) |
| DVC pipeline (`dvc.yaml`, `params.yaml`) | yes | none | gap (M2, optional) |
| `scripts/fetch.sh` pinned download | yes | none | gap (M2) |
| Browser | `slm_wasm` CPU in WASM | WebGPU in the tab | different (Phase 5b) |
| GUI | Tauri desktop app | local web GUI | different; feature check in M8 |
| RAM guard (Linux), `--force` | yes | GPU buffer pre-flight and `--gpu-mem-mb` | different |
| Determinism | bit-identical across thread counts | bit-identical across repeat runs per adapter | different (stated in ARCHITECTURE.md) |
| Device choice | CPU only | `--device cpu\|gpu\|auto` | GPU-only feature |
