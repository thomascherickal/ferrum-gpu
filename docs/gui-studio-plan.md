# GUI Studio plan: parity with ferrum-cpu's Studio, plus GPU-specific tabs

Written 2026-11-17. The goal: the ferrum-gpu web GUI offers what ferrum-cpu's Studio offers (read from
`ferrum-cpu/ferrum_gui`, never modified) and adds what only a GPU engine can. Execution is autonomous;
each phase is one commit on `worktree-gui-studio` with tests, and the branch is merged to `main` at
the milestones below after the full gate (fmt, clippy, four test configurations, time limits,
coverage floor, `cargo deny`).

## What ferrum-cpu's Studio has (the parity target)

Twelve tabs: **Datasets, Prepare, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, Serve,
System, Capable**; a Beginner/Expert mode (advanced tabs hidden for beginners); a tool-tip on every control
(at least 30 words, mode-aware); live time and memory estimates; settings that persist; keyboard-operable
tabs and ARIA roles; streamed progress, logs and generated text; cancel with a clean stop.

What ferrum-gpu has today: four tabs (Device, Model, Train, Generate) for the byte-level Llama-style
model only. The GPT-2 path (BPE tokenizer, `--tie`, `--dropout`, validation with early stopping,
fine-tuning, eval, tokenize, exports) exists only in the CLI.

## Architecture decision: the GUI drives the CLI as jobs

`ferrum_gpu_cli` depends on `ferrum_gpu_gui` (for the engine layer), so the GUI cannot call the CLI's
code. Rather than move 3,000 lines between crates, the GUI server runs `ferrum-gpu` and
`ferrum-gpu-server` as **child processes ("jobs")**: it streams their output to the page, parses the
`--verbose` progress lines, and stops them cleanly (the `<out>.stop` file, then `Child::kill` on a
positive pid; rule 16). One source of truth for every command, and a GPU fault ends the job, not the GUI.
Costs: a binary must be found (next to the GUI executable, `--cli PATH`, or `PATH`), and the progress
format is a contract (tested with a stub CLI). Read-only work that needs no engine run (GGUF inspection,
memory estimates, the device table, the hardware formulas) is done in-process by the GUI crate.

## Phases

| # | Phase | Deliverable | Tests |
|---|---|---|---|
| 1 | **Studio shell** (done) | ARIA tablist with arrow/Home/End keys that skips hidden tabs, Beginner/Expert mode (advanced tabs hide for beginners), theme, persisted settings, mode-aware tool-tips with a help entry per control (expert text 25+ words, beginner 15+); the existing four tabs run inside it, and each later phase adds its tab, its help entries and its `data-adv` mark | 66 node checks; Rust: every control has help, none is stale, texts are real, modules are served; the Chrome UI test still passes |
| 2 | **Jobs backend** | `/api/jobs` start, poll (`since`), stop, list; CLI discovery; line streaming; progress parsing; no arbitrary commands (an allow-list of subcommands and flags) | stub-CLI unit tests; integration with the real binary when built |
| 3 | **Datasets, Prepare, Tokenize** | Browse server-side folders (read-only listing), prepare a corpus, tokenize with a tokenizer file, token and byte counts | listing refusals, path handling, stub jobs |
| 4 | **GPT-2 Train** | BPE training with vocab, tie, dropout, QAT, validation share and patience, checkpoint and resume, stop; live loss and validation chart from the job's progress lines; device, adapter and GPU budget fields | parse tests, stub run, real tiny run |
| 5 | **Fine-tune, Evaluate, Generate** | `finetune-gguf`, `eval` (loss and perplexity), `generate` for GPT-2 files with top-k, top-p, repetition penalty, streamed text | stub and real runs |
| 6 | **GGUF and Models** | Inspect a GGUF in-process (header, metadata, tensors, shapes, quantisation, size), list models in a folder with shape and parameter count, load into the Model tab | fixtures, hostile files |
| 7 | **Export and Serve** | `export-gguf` (f32, int8, int4), `export-ollama`, `export-lmstudio` with tool detection; start and stop `ferrum-gpu-server` with port and token, live log | stub tools, real server start |
| 8 | **System and Capable** | Adapters, drivers, RAM; what fits: pick memory and precision, see the largest model (the formulas of HARDWARE-CAPs.md, implemented in Rust and tested against the script's table) | table equality test |
| 9 | **GPU lab (GPU-specific)** | Adapter picker with notes, **VRAM budget** (planned memory against the card's free memory, read through an injected `nvidia-smi` reader; fixes the pre-flight finding in CAPABILITY.md), CPU-vs-GPU speed check, precision comparison (f32, Q8_0, Q4_0 size and tokens/s), the WebGPU self-test and benchmark moved here | injected readers, plan arithmetic, fake adapter lists |
| 10 | **Polish and docs** | Accessibility pass, empty and error states, `docs/gui.md` rewritten, README GUI section, coverage numbers | a11y contract test, docs-versus-routes test |

Milestones (merge to `main`): after phase 3 (shell, jobs, data), after phase 7 (train through serve), after
phase 10 (everything).

## Rules that apply throughout

- Tests first, one concern per file, a `tests/` sibling per module (rule 11); the suite passes with no GPU
  and without `node`, `nvidia-smi`, `ollama`, `lms` (those tests skip with a printed reason).
- No unsafe, no outside dependencies; JavaScript is vanilla and loaded from the embedded `/web/` tree.
- The server stays loopback-only; job commands come from an allow-list, never from a free-form string; paths
  are passed as arguments, never through a shell.
- Every number shown in the UI that is an estimate says so; no invented figures in docs.
- Not in scope: multi-user access, authentication on the GUI, a Tauri desktop wrapper.
