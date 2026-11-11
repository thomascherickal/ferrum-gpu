# Golden fixtures

Data files that pin behaviour copied from ferrum-cpu, so tests here never read outside this repository (CLAUDE.md, independence rule). Each fixture is produced once by running a ferrum-cpu binary in a scratch directory, then copied here and listed below. After that it is ours.

Fixtures so far were all made on 2026-11-11 with ferrum-cpu's `train_transformer` release binary, built from commit `0bb0f92b7e71969f11a49bd80f346140c45f2983` into a scratch target directory (ferrum-cpu itself untouched). The tokenizer source was copied from the same commit (provenance headers in `ferrum_tok/src`).

## Ledger

| file | what | command (run in a scratch directory containing `corpus.txt`) | ferrum-cpu commit | sha256 |
|---|---|---|---|---|
| `bpe/corpus.txt` | 2010-byte, 31-document corpus (ASCII, German, Japanese, emoji, Cyrillic, tabs, newlines) written for this repository, documents joined with `<|endoftext|>` | (ours; not from ferrum-cpu) | n/a | c64a8f054e12291dd305f869144c1edcc10d20489e1c336d65d2bdea79948a56 |
| `bpe/tok.gguf` | standalone tokenizer file, vocab 320 | `train_transformer train corpus.txt model.gguf --preset tiny --vocab 320 --epochs 1 --tokenizer tok.gguf --seed 1` | 0bb0f92b7e71969f11a49bd80f346140c45f2983 | ca035b85db47168a4db57cff2d818fd0078954949f17b30e126c38892be653b7 |
| `bpe/model.gguf` | the GPT-2 model that run wrote: 181440 bytes, int8, vocab 320, context 64, BPE metadata inside | same command | 0bb0f92b7e71969f11a49bd80f346140c45f2983 | 623af38b770ace7617725682bf84c0bb212d62ab78be1784e6b9775928af36ec |
| `bpe/corpus.tok.gguf` | token file: 942 ids, 31 documents, I16 | `train_transformer tokenize corpus.txt --tokenizer tok.gguf --out corpus.tok.gguf` | 0bb0f92b7e71969f11a49bd80f346140c45f2983 | 19a90b2726ca43fa65db2e3b8ab710a72b0add3eb064a1beb1ee30efb8d75c44 |

Used by: `ferrum_model/src/tokenizer_gguf/tests.rs` (tokenizer file, model metadata and ids equal ferrum-cpu's; our GPT-2 reader loads `model.gguf`).

`scripts/make-fixtures.sh` holds the commands for each row and prints the ledger line to paste here.
