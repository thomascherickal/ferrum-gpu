# GGUF support

ferrum-gpu reads and writes GGUF **v3**. The container code is in `ferrum_gguf`; the model mappings are in
`ferrum_model` (`gguf/gpt2.rs`, `gguf/llama/`). The container follows the public GGUF/ggml specification.

## The container (`ferrum_gguf`)

- **Parser:** `Gguf::parse(bytes)` is safe against hostile input: every read is bounds-checked, sizes use
  checked arithmetic, and nothing is allocated from a header number before it is shown to fit in the file.
  Malformed files are `GgufError` values, never panics.
- **Element types read:** F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_1 (blocks of 32) and the k-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (super-blocks of 256). The block types other than Q8_0/Q4_0 are **read only**: they are dequantized to f32 on load (formulas from ggml, cross-checked against ferrum-cpu's dequantizers; unit-tested on hand-computed blocks, not yet against a real downloaded file, because none is kept in this repository) and can be re-written with `export-gguf`. Integer types I16 and I32 are accepted as raw bytes for token files. Any other ggml tensor type (for example Q8_K, IQ formats, BF16) is refused with `unsupported ggml tensor type N`.
- **Element types written:** F32, Q8_0, Q4_0.
- **Writer:** `GgufBuilder` builds files for exports, fixtures and checkpoints. Default alignment is 32.
- **Codecs:** `quant::{decode, encode}` and `f16`. Encoding rounds half away from zero and stores the block
  scale as an f16, so `encode(decode(x))` is value-idempotent (the bytes can differ for blocks whose f16
  scale underflows).

## Layout convention

GGUF stores each matrix with ggml dimension 0 (contiguous) as the **input** dimension, so a stored matrix is
`[out, in]` row-major. The in-memory models here hold weights as `[in, out]`, so loading and saving
transpose once each way. Vectors (norm gains, biases) are always F32 on disk.

## `gpt2`

`general.architecture = "gpt2"`, metadata under `gpt2.*` (including `gpt2.attention.layer_norm_epsilon`).
`attn_qkv.weight` stacks Q, K and V (each transposed) along the output axis. A missing `output.weight`
means the head is tied to `token_embd.weight`. Written by `ferrum_model::gpt2_gguf::to_gguf(model, Quant)`,
read by `from_gguf`.

## `llama` and `qwen2`

Read by `ferrum_model::llama_gguf::from_gguf`, written by `to_gguf(model, Quant)`.

| Aspect | Behaviour |
|---|---|
| Architecture | `llama` pairs with RoPE style NORM (adjacent pairs); `qwen2` pairs with NEOX (split halves) |
| Tensors | `token_embd.weight`, `blk.N.*` (attention q/k/v/output, FFN gate/up/down, norms), `output_norm.weight`, optional `output.weight` |
| Head | Missing `output.weight` means tied to `token_embd.weight` |
| Biases | Accepted on attention q, k and v only. A bias anywhere else is **refused**, not dropped |
| Grouped-query attention | `attention.head_count_kv` smaller than `head_count`, with `attention.key_length` for the head size |
| RoPE | `rope.freq_base`, `rope.dimension_count` (partial rotary supported) |
| RoPE scaling | Supported: none, `linear` (`rope.scaling.factor` or `scale_linear`) and a `rope_freqs.weight` tensor. **Refused:** `yarn`, `longrope`, and any `attn_factor` other than 1, because running them unscaled would be silently wrong |
| Quantisation on write | `Quant::F32`, `Q8_0` or `Q4_0`; matrices whose input dimension is not a multiple of 32 stay F32 |

Anything outside this table is reported as an error that names the problem; a model that loads is a model
that runs correctly.

## Tokens

The files this project writes carry **no tokenizer**. The CLI, server and GUI use a byte tokenizer
(vocabulary 256). A GGUF with a different vocabulary loads and runs at the token level (for example through
`GpuLlama`), but the CLI cannot turn text into its tokens. A BPE tokenizer is not implemented yet.

## Training checkpoints

`ferrum_model::checkpoint` stores a resumable run as a GGUF file: weights, both Adam moments (`p{i}.data`,
`p{i}.m`, `p{i}.v` as F32 in canonical parameter order), and the loop position under `ferrum.ckpt.*`
metadata (step, corpus id and so on). `atomic_write` writes a temporary file, flushes and renames it over the
target, so a crash leaves the old file or the new one, never half of one.

## Compatibility with ferrum-cpu

The intent is that a model trained on either side loads on the other through these files. **That has not
been verified with a real file.** The repository contains no stored `.gguf` fixture: tests generate files
with this project's own writer, and the tensor names, metadata keys and layout rules were checked against
ferrum-cpu's *source* conventions (commit 56a9d3a), not against a file ferrum-cpu wrote. Adding a stored
fixture produced by ferrum-cpu is an open item (see `docs/plans/PHASE-3.md`, task 3.1).
