# MATCHER-PLAN: reaching feature parity with ferrum-cpu

**Goal:** everything a user can do with ferrum-cpu (train, fine-tune, run, score, serve, export small causal Transformers as GGUF), they can do with ferrum-gpu, on any GPU or on the CPU, with the same files going both ways.

**Status:** Phase 7 (device matrix) is complete for the hardware available. M0 (audit) is complete, against ferrum-cpu commit `56a9d3ada59df2a7f3f4e128d04a96b59211581c`; the ledger is [docs/PARITY.md](docs/PARITY.md). M1 is next.

## 0. The independence rule (CRITICAL, unchanged)

This plan changes nothing about [CLAUDE.md](CLAUDE.md). Parity is reached by **re-implementing and copying in**, never by linking.

1. No `path =` or `git =` dependency on ferrum-cpu or any of its crates, and no dependency pointing outside this repository. `tests/tests/test_independence.rs` keeps enforcing it; it must stay green at every commit.
2. No `include_str!`, `include_bytes!`, `build.rs` or test that reads outside this repository. Nothing under `../ferrum-cpu`, ever, from code or tests.
3. **What we may do with ferrum-cpu:** read its source, docs and `--help` output while planning or porting (CLAUDE.md rule 6), and run its binaries with output sent to scratch to *generate golden data*.
4. **Copying code in:** when a phase starts, copy the file(s) it needs, each with a provenance header (source path, ferrum-cpu commit hash). After the copy it is ours and is not kept in sync. Do not bulk-copy up front.
5. **Cross-compatibility is proven through files, not code.** Golden fixtures (`fixtures/*.gguf`, token files, expected-output JSON) are produced by a ferrum-cpu binary run in scratch, then copied into `fixtures/` with a `fixtures/README.md` entry (command, ferrum-cpu commit, date, sha256). Our tests read only `fixtures/`. The fixture is data, so it does not create a dependency; losing ferrum-cpu changes nothing about our build or tests.
6. ferrum-cpu is never modified. The reverse direction (ferrum-cpu loading a file we wrote) is checked by hand, and the result is recorded in the phase doc with the command and the ferrum-cpu commit; it is not an automated test here.
7. Dependencies we add must be justified by a measurement or by necessity, be pure-safe-Rust to call, and pass `cargo deny`. Prefer `std` and hand-written code, as ferrum-cpu does.
8. Every other rule in CLAUDE.md stands: tests first, `#![forbid(unsafe_code)]`, no invented numbers, files under about 500 lines, one feature per commit, nothing merged or pushed to `main` unless asked.

## 1. Method

Each M-phase starts with an audit step: read the ferrum-cpu source for that area (`ferrum_core/src/...`, `slm_cli/src/...`, `slm_server/src/...`), list exact behaviours and flags in the phase doc, then write tests from that list (and from golden fixtures) before any code. The gap table below was built from ferrum-cpu's README, STATUS, `--help` text and source tree names; **it is a planning list, not an audit.** Each row is confirmed or corrected in its phase before work starts.

Parity is measured three ways, never assumed:

- **Behaviour:** same flag, same default, same result on the same input (golden fixtures).
- **File:** a file written by one side loads on the other and gives the same greedy continuation (fixtures one way, manual check the other, per rule 6 above).
- **Numbers:** loss, perplexity and logits agree within the tolerances already defined (EXACT 1e-6, REDUCING/BACKWARD/FORWARD 1e-5), CPU oracle versus GPU.

Where GPU and CPU differ for good reason (the `auto` device rule, no ferrum-cpu RAM guard on GPU, GPU memory pre-flight instead), the difference is written down in `docs/PARITY.md` (created in M0), not hidden.

## 2. Gap table (ferrum-gpu today versus ferrum-cpu)

> **Superseded by [docs/PARITY.md](docs/PARITY.md)** (M0). The audit changed three things below: (1) `ferrum-gpu train` writes a Llama-style model while ferrum-cpu writes a GPT-2 model, so M4 adds `train --arch gpt2|llama` with `gpt2` as the default; (2) presets are different models under the same names; (3) ferrum-cpu reads Q4_1, Q5_0, Q5_1, Q8_1 and Q2_K to Q6_K (11 block formats against our 3), and its server also has `/v1/loglikelihood_rolling` and a 32-request in-flight cap. The table below is kept as written for history.

"Have" is from this repository's code. "Gap" rows get a phase.

| Area | ferrum-cpu | ferrum-gpu today | Phase |
|---|---|---|---|
| Tokenizer | Byte-level BPE with GPT-2 pre-tokenizer, trained and saved as GGUF; character-level (`--vocab 0`); `<|endoftext|>` reserved | Byte tokenizer only (vocab 256) | M1 |
| GGUF tokenizer metadata | Read and write, so llama.cpp tokenizes the same way | Not read or written | M1 |
| Quant formats in | F32, F16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K to Q6_K | F32, F16, Q8_0, Q4_0 | M3 |
| Quant formats out | int8 (Q8_0), int4 (Q4_0), f32 | q8_0, q4_0, f32 | have |
| Corpus tools | `prepare` (clean, dedup, NFKC, split, manifest), `tokenize` (token files, streamed from disk) | None | M2 |
| Train options | `--epochs`, `--tokens`, `--stride`, `--val`/`--patience`, `--max-seconds`, stop file, `--metrics-csv`, `--verbose` ETA, `--force`, `--sample`, presets | `--steps`, `--lr`, `--batch`, `--context`, `--seed`, `--preset`; the engine has weight decay, dropout, cosine/warmup, clip, tying, resume, but the CLI exposes few | M4 |
| Checkpoint and resume | GGUF checkpoint `.fckp`, atomic writes, refuses tokenizer or context mismatch | Checkpoint format and resume exist in the engine; atomic writes and mismatch refusal to be checked | M4 |
| Generation options | `--top-k`, `--top-p`, `--repetition-penalty`, `--temp`, `--stop` (repeatable), `--max-tokens`, `--stream`, `--gen-seed` | Sampling has temp/top-k/top-p/repetition; no stop strings, no streaming, flags partly exposed | M5 |
| Imported models | `run-gguf` with chat template, `--ids`, `--raw`, `--quant`, `--resume`; `eval-gguf` with `--ctx`, `--quant`, `--json`; `finetune-gguf`; `export-gguf` | `run-gguf`, `finetune-gguf`, `eval` exist; no chat template, no export-gguf, no `--json` | M3, M5 |
| Export to other runtimes | `export-ollama`, `export-lmstudio` | None | M6 |
| Server | axum: log-likelihood, rolling log-likelihood, generation; imported models; bearer token (`FERRUM_TOKEN`), non-loopback needs a token; `--max-batch`, `--max-chars`, `--max-body`, `--max-tokens`, cancel | std-only: `/health`, `/v1/info`, `/v1/generate`, `/v1/loglikelihood`; loopback only, no token | M7 |
| Evaluation | LightEval module (`FerrumModel`, custom tasks) | None | M7 |
| Data ops | DVC pipeline (fetch, prepare, train, eval), `scripts/fetch.sh` | None | M2 |
| Browser | `slm_wasm` runs a model on the CPU in the browser | WebGPU inference in the browser tab (Phase 5b) | have (different route) |
| GUI | Tauri desktop app | Local web GUI | have (different shell; feature check in M8) |
| Memory | RAM guard (Linux), `--force` | GPU buffer pre-flight, `--gpu-mem-mb`; no VRAM discovery wired | M4 |
| Determinism | Bit-identical across thread counts | Bit-identical across repeat runs per adapter | have |

## 3. Phases

Each phase ends at a gate. A failed gate is recorded with real numbers and the consequence, and work continues with the recommended action (CLAUDE.md rule 9).

### M0: audit and parity ledger (done 2026-11-11)

- Read the ferrum-cpu CLI, server and GGUF code named in the table; correct the table.
- Create `docs/PARITY.md`: one row per feature, status (have / partial / gap / deliberately different), with the reason for any difference.
- Create `fixtures/README.md` (empty ledger) and a `scripts/make-fixtures.sh` that *documents* the ferrum-cpu commands to run in scratch; it never runs from tests.
- **Gate (met):** rows confirmed against source where source was read, otherwise marked "doc" and re-read in their phase; commit hash recorded in PARITY.md.

### M1: BPE tokenizer and GGUF tokenizer metadata (library part done 2026-11-11)

Done: new crate `ferrum_tok` (copied from ferrum-cpu `0bb0f92`, depends only on `unicode-general-category`, cargo deny ok), `ferrum_model::tokenizer_gguf` (tokenizer file and model metadata, both directions), integer tensor types in the GGUF reader, golden fixtures in `fixtures/bpe/` (tokenizer file, model file, token file, corpus). Results: tokenizer file, metadata and the 942 corpus ids equal ferrum-cpu's; our GPT-2 reader loads ferrum-cpu's `model.gguf`. Not done and moved: character-level mode (`--vocab 0`) goes with M4 (it is a model-type change, not a tokenizer one); using BPE in `train`/`generate`/the server goes with M4/M5.

- Copy in the byte-level BPE trainer, encoder/decoder and GPT-2 pre-tokenizer (provenance headers), plus the Unicode category and normalization it needs. Decide, by a short measurement, between a small hand-written table and the two pure-Rust crates ferrum-cpu uses; either is acceptable under the rule, a dependency needs a `cargo deny` pass.
- Read and write `tokenizer.ggml.*` metadata in our GGUF layer; keep byte mode for old files.
- Character-level mode (`--vocab 0`).
- Tests: round trip on adversarial UTF-8; merges identical to a fixture trained by ferrum-cpu on a fixed corpus; llama.cpp-style prompts tokenize to the fixture's ids.
- **Gate:** tokenizer output equals the fixtures byte for byte; our GGUF with BPE metadata carries a fixture-checked token list.

### M2: corpus tools and data pipeline (tokenize and token files done 2026-11-11; `prepare` not done)

Done: `tokenize` (file or directory), token-file writer and streaming reader (`ferrum_model::token_file`, byte-identical to ferrum-cpu's file on the fixture), SHA-256 in std, `TokenSource` so the training loop reads windows from disk, `train` on a token file (equal to training from the text, byte for byte). Not done: `prepare` (cleaning, NFKC, Gutenberg strip, dedup, split, manifest; it needs ~800 lines ported from `dataset.rs`, `prepare/` and a JSON/NFKC decision), `scripts/fetch.sh`, a DVC pipeline.

- `ferrum-gpu prepare` (clean, optional NFKC and lowercase, Gutenberg strip, dedup, split, hash manifest) and `tokenize` (token file, 2 bytes per token, streamed windows from disk).
- `train` accepts a token file and streams windows without loading the corpus.
- A DVC-style `params.yaml` pipeline is a convenience; ship a plain shell script first, DVC files only if wanted.
- **Gate:** `prepare` output and manifest equal the fixtures on a small corpus; training from a token file gives the same loss curve as from text on the same tokens.

### M3: model formats and quantization (done 2026-11-11)

Done: Q4_1, Q5_0, Q5_1, Q8_1 and Q2_K to Q6_K read paths (`ferrum_gguf::blocks`), `export-gguf`. Honest limits: the formulas are checked on hand-computed blocks that follow ferrum-cpu's tests and ggml's layout, not on a downloaded real K-quant model (ferrum-cpu cannot write these formats, so there is no golden file to copy); everything is dequantized to f32 on load, so memory is f32-sized; nothing is written in these formats. Not met as written: the gate asked for a decode of a fixture model to the same greedy tokens as ferrum-cpu; no such fixture could be produced.

- Add Q4_1, Q5_0, Q5_1, Q8_1 first (simple blocks), then the K-quants Q2_K to Q6_K, as dequantizers on the CPU oracle first and then in WGSL. Each format gets a golden block fixture.
- `export-gguf` (re-quantize an imported model to a chosen format).
- **Gate:** every listed format loads, dequantizes to the fixture values, and decodes a fixture model to the same greedy tokens as the recorded ferrum-cpu output.

### M4: training parity (core done 2026-11-11)

Done: `train --arch gpt2` (default) with BPE, ferrum-cpu's presets and size flags, epochs/steps/tokens budgets, stride, weight decay, cosine and warmup, clip, QAT on by default, `--quant int8/int4/f32`, checkpoint and resume, `--val`/`--patience`, `--max-seconds` and stop file, `--metrics-csv`, `--verbose`, `--sample`, `finetune-gguf` for GPT-2 files, and the GPU runs the same loop (loss curve identical to four decimals on the MX330). Our `train` output loads in ferrum-cpu (manual check, docs/cli.md). Left in M4: `--tie` and `--dropout` (need engine work, refused by name today), `--vocab 0` character-level, token-file input (M2), VRAM discovery for the pre-flight, presets for the Llama path unchanged. Also found: the server only loads Llama-style files; serving GPT-2 files moves to M7.

- Expose every engine option the CPU CLI has: epochs and `--tokens` budget, stride with random phase, `--val`/`--patience` (best-epoch keep, early stop), `--max-seconds` and `<model>.stop`, `--metrics-csv`, `--verbose` with tokens/s, grad norm and ETA, `--tie`, `--no-qat`, `--weight_decay`, `--dropout`, `--cosine`, `--warmup`, `--clip`, `--sample`.
- Presets: check ours against ferrum-cpu's tiny/small/medium and record any difference.
- Checkpoint: atomic writes, refuse resume on vocabulary or context mismatch, same canonical parameter order as the checkpoint format already used.
- Wire VRAM discovery (`budget::resolve`) into the pre-flight.
- **Gate:** the same corpus, seed and options give a loss curve within tolerance of the ferrum-cpu fixture curve (recorded, not re-run from tests), and a killed run resumes at the right step.

### M5: generation and scoring parity

- `--stop` (repeatable), `--max-tokens`, `--stream`, chat-template formatting for instruct GGUFs with `--raw` to turn it off, `--ids`, `--resume` overlay.
- `eval` and `eval-gguf` options: `--ctx`, `--quant`, `--json`, rolling log-likelihood.
- Decide where scoring runs: keep it on the CPU oracle unless a measurement shows a GPU win.
- **Gate:** for fixtures, the continuation, perplexity and log-likelihood equal the recorded ferrum-cpu values (greedy: exact tokens; scores: within 1e-5).

### M6: export to other runtimes

- `export-ollama` (Modelfile, and the `ollama create` call behind a flag) and `export-lmstudio`.
- **Gate:** generated Modelfile and metadata equal the fixtures; the manual check that llama.cpp loads our GGUF is recorded in the phase doc with command and date.

### M7: server and evaluation parity (server done 2026-11-11)

Done: ferrum-cpu's route shapes and limits, `/v1/loglikelihood_rolling`, GPT-2 files with BPE, bearer token and exposure rule, in-flight cap, stop strings, top-p and repetition penalty. Left: request cancel on disconnect, `--max-batch` semantics beyond the request count, the Python evaluation adapter (the existing LightEval module has not been tried against this server).

- Bearer-token auth (`FERRUM_TOKEN`), non-loopback bind only with a token, `--allow-unauthenticated` as an explicit opt-out; `--max-batch`, `--max-chars`, `--max-body`, `--max-tokens`; request cancel; rolling log-likelihood route; imported-model routes.
- A small Python evaluation adapter in `eval/` that talks to our server over HTTP (no code shared with ferrum-cpu's LightEval module; written fresh against the same route shapes).
- **Gate:** the server route tests plus an end-to-end run through the real binary, including the auth refusals.

### M8: GUI and docs parity, final matrix

- Compare the GUI's tabs and options to ferrum-cpu's GUI manual section; add what is missing (BPE, validation split, checkpoints, stop strings).
- Update README, USER-GUIDE, GGUF.md, `docs/PARITY.md` to all-green or an honest list of remaining differences.
- Re-measure coverage with the new crates and record it in `docs/COVERAGE.md`.
- **Gate:** `docs/PARITY.md` has no "gap" rows except those marked "deliberately different" with a reason.

## 4. Order and dependencies between phases

M0 first. M1 before M2, M4 (BPE vocab sizes) and M5 (chat templates need tokenizer metadata). M3 is independent of M1 and may run alongside it. M6 needs M1 and M3. M7 needs M5. M8 last. Phase 7 (device matrix) runs before M0 and again after M8, because new kernels (K-quants) change what each adapter row means.

## 5. Risks

- **Scope.** K-quants, BPE, and the data pipeline are each a large job; the order puts correctness-critical pieces (tokenizer, formats) first.
- **Fixtures go stale.** ferrum-cpu is still moving; every fixture records its source commit, and a mismatch later is read as "ferrum-cpu changed" until proven otherwise.
- **Unicode tables.** Writing them by hand is long; the crate route needs a `cargo deny` pass and a justification. Decided in M1.
- **GPU can lose.** Where a GPU path is not faster (decode, scoring), the `auto` rule keeps the CPU path. Parity here means the feature exists and gives the same answer, not that it runs on the GPU.
- **No second machine for the reverse check.** Loading our files in ferrum-cpu or llama.cpp is a manual step recorded per phase.

## 6. Not planned

Anything ferrum-cpu has explicitly moved out (`ferrum-ml` tabular models, the archived features in `ferrum-junk`), per-vendor backends, f16 mixed precision, multi-GPU, and a Tauri desktop shell (the local web GUI stays).
