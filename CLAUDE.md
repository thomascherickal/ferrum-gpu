# ferrum-gpu

A GPU engine (wgpu / WGSL, any GPU incl. Apple Silicon) for small causal
Transformers, GGUF in and out. Plan: `PLAN.md` (index) and `docs/plans/PHASE-*.md`.

## Independence rule (strict, user decision 2026-11-10)

**ferrum-gpu is 100% independent of ferrum-cpu.** Duplicate code if necessary.

- No `path =` or `git =` dependency on ferrum-cpu or any of its crates, and no
  dependency that points outside this repository.
- No `include_str!` / `include_bytes!` / `build.rs` / test that reads a file
  outside this repository (nothing under `../ferrum-cpu`).
- Code we need from ferrum-cpu is **copied in**, when the phase that needs it
  starts (not bulk-copied up front). Each copied file starts with a provenance
  header: source path and the ferrum-cpu commit hash it came from. After the
  copy it is ours; do not try to keep it in sync.
- Cross-loading of models ("trained on either side loads on the other") is
  proven through the GGUF format with fixture `.gguf` files stored in this repo.
- ferrum-cpu is never modified for GPU work.
- `tests/tests/test_independence.rs` enforces this; keep it green.

## Working rules

1. Tests first; every added file gets unit tests. Tiers: unit per file,
   functional per feature, integration per module, regression with every fix.
2. External tools and hardware (`nvidia-smi`, sysfs, GPU adapters, vulkaninfo)
   are reached through injected inputs: parse fixtures, use temp-dir trees,
   fake adapter lists. Tests needing a real adapter skip with a printed reason;
   the suite must pass on a machine with no GPU.
3. One feature per commit, Conventional subjects, full suite + clippy
   `-D warnings` + fmt green before each commit. Default branch is `main`.
4. `#![forbid(unsafe_code)]` in every crate root. Never call a dependency's
   `unsafe fn`.
5. No invented numbers: every figure in docs comes from a real run (adapter,
   driver, backend, OS, wgpu version, date). Say what was not run.
6. Stay inside the project directory. No installs needing sudo without asking.
7. Low-memory box (15 GB): `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off
   CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`, one cargo command at a time.
8. Deliberate shortcuts carry a `// ponytail:` comment naming ceiling and upgrade.
9. Each phase ends at its gate. If a gate fails or misses, record the real numbers and the
   consequence in the phase doc, then continue with the recommended action (see rule 10)
   instead of halting; stop only for a real blocker (rule 10).
10. **Stay autonomous (user decision 2026-11-10).** Whenever there is a question about how to
    proceed, continue with the **Recommended** option without asking. Make routine decisions
    by the model's own judgement and say in one line what was decided. Do not exit autonomous
    mode to ask for confirmation. Ask the user only when there is no recommended option and no
    sensible default, or for a hard blocker (needs credentials, sudo, or an action outside
    this repository). The existing limits still apply: never invent numbers, never touch
    ferrum-cpu, never force-push, and never push or merge to `main` unless the user asks.
