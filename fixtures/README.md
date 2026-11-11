# Golden fixtures

Data files that pin behaviour copied from ferrum-cpu, so tests here never read outside this repository (CLAUDE.md, independence rule). Each fixture is produced once by running a ferrum-cpu binary in a scratch directory, then copied here and listed below. After that it is ours.

No fixture exists yet. M1 adds the first (BPE tokenizer), M3 the quantization blocks, M4 and M5 the training and generation references.

## Ledger

| file | what | produced by (command, run in scratch) | ferrum-cpu commit | date | sha256 |
|---|---|---|---|---|---|
| (none yet) | | | | | |

`scripts/make-fixtures.sh` holds the commands for each row and prints the ledger line to paste here.
