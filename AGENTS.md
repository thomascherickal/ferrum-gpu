# AGENTS.md

Project-wide rules and automation standards for all coding agents and contributors.

## 1. Testing & Coverage

- **100% Test & Code Coverage**: Maintain 100% test pass rate and 100% code coverage before any commit.
- **Core Test Scope**: Tests must cover:
  - Functional scenarios
  - Edge-case scenarios
  - Regression scenarios
- **Expanded Test Categories** (introduce and maintain as the project grows):
  - **Unit tests** (isolated component/function behavior)
  - **Integration tests** (interactions between modules, services, and data layers)
  - **Smoke tests** (fast sanity checks verifying critical paths work after builds/deployments)
  - **End-to-end (E2E) tests** (full user-facing workflows across the entire stack)
  - **Performance & load tests** (latency, throughput, and behavior under expected/peak load)
  - **Security tests** (authentication, authorization, input validation, and vulnerability checks)
- **Test Suite Runtime Limit**: Optimize the complete test suite so that it never takes more than **30 minutes** to run.
- **Live Test Timing**: Maintain `test-time-report.md` in every repository to record live execution times for all tests and suites, flagging slow tests for optimization.
- **Commit Gate**: Enforce a commit/CI gate that blocks commits and merges if any test fails, if code coverage drops below 100%, or if the test suite exceeds the 30-minute limit.

## 2. Code Quality, Modularity & Optimization

- **File Size Limit**: No source code or test file may exceed **200 lines**. Split larger files into smaller, coherent modules.
- **Design Principles**: Strictly enforce **DRY** (Don't Repeat Yourself) and **SOLID** principles.
- **Duplicate Code Detection**: Continuously scan for and eliminate duplicate code.
- **Per-Commit Code Optimization**: On every commit, review and optimize the code to check whether it can run faster and remove any unnecessary, redundant, or dead blocks of code.
- **Language-Specific Tooling**: Suggest, install, and configure appropriate linters, formatters, type checkers, and static analysis tools for the project's language(s).

## 3. Resource Management & Performance

- **CPU & GPU Profiling**: Profile CPU and GPU usage during builds, test runs, and runtime execution to detect bottlenecks and prevent system overload.
- **Profile Reporting**: Store and keep CPU and GPU profile results up to date in `system-profile.md`.

## 4. Documentation & Project Files

Every project must include and keep up to date (after every push):

- `readme.md` (both full documentation and a quick-start guide)
- `next-steps.md` (reusable post-commit/push checklist)
- `metrics.md` (project health, coverage, performance, and quality metrics)
- `test-time-report.md` (live timing report tracking how long all tests take to run)
- `system-profile.md` (CPU and GPU profiling results and resource-usage trends)

> **Naming Convention**: Name all Markdown files in lowercase `kebab-case` (e.g., `readme.md`, `next-steps.md`, `metrics.md`, `test-time-report.md`, `system-profile.md`), keeping uppercase only where strictly required by external tools (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `SKILL.md`).

## 5. Licensing

- **Default License**: **AGPL-v3**.
- **Terms**: Free for internal use; commercial use requires a separate commercial license from **Thomas Cherickal**.

## 6. Automation

- Automate all of the above—testing across all applicable categories, 100% coverage verification, 30-minute test-suite runtime enforcement, live `test-time-report.md` updates, 200-line file-length checks, duplicate-code detection, per-commit optimization checks, linting/formatting, CPU/GPU profiling (`system-profile.md`), commit gates, and documentation/metrics updates—via Git hooks and CI/CD pipelines.

## 7. Post-Commit / Post-Push Checklist (`next-steps.md`)

Run the following steps after each commit/push:

1. Verify all functional, edge-case, regression, and applicable expanded tests (unit, integration, smoke, E2E, performance/load, security) pass.
2. Confirm 100% code coverage with zero regressions.
3. Verify the complete test suite runs in under 30 minutes and update `test-time-report.md` with live timings.
4. Verify no code or test file exceeds 200 lines.
5. Run duplicate-code detection, review for per-commit performance optimizations, and remove any unnecessary or dead code blocks.
6. Run language-specific linters, formatters, and static analysis checks (DRY/SOLID compliance).
7. Profile CPU and GPU usage, verify no resource overload or performance regressions, and update `system-profile.md`.
8. Confirm the commit/CI gate passed cleanly.
9. Update `metrics.md` with the latest test, coverage, and performance metrics.
10. Update `readme.md` (full docs and quick-start) to reflect any changes.
11. Review and update `next-steps.md` with any follow-up actions for the next iteration.
12. After all that is done, commit or push the changes to the repository.

---

## Project-Specific Rules & Context

# ferrum-gpu

A GPU engine (wgpu / WGSL, any GPU incl. Apple Silicon) for small causal
Transformers, GGUF in and out. Plan: `plan.md` (index) and `docs/plans/phase-*.md`.

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
6. Writes stay inside the project directory (plus the session scratch directory). Reading
   outside the sandbox is allowed when it is needed (for example the cargo registry, system GPU
   info, or ferrum-cpu's source or binaries to compare against, with build output sent to
   scratch); never modify anything outside the repo, and never modify ferrum-cpu. No installs
   needing sudo without asking.
7. Low-memory box (15 GB): `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off
   CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`, one cargo command at a time.
8. Deliberate shortcuts carry a `// ponytail:` comment naming ceiling and upgrade.
9. Each phase ends at its gate. If a gate fails or misses, record the real numbers and the
   consequence in the phase doc, then continue with the recommended action (see rule 10)
   instead of halting; stop only for a real blocker (rule 10).
11. **Modular code, tests for everything, coverage toward 100% (user decision 2026-11-10).**
    Keep each file to one concern with a small public surface (split a file that mixes
    concerns or grows past roughly 500 lines of non-test code; move big test modules into a
    sibling `tests.rs`, and give a command with several concerns its own directory: `mod.rs` for the
    command, one file per concern (flag tables, argument splitting, ...), and `tests/` or a `tests.rs`
    beside each file, one test file per concern, with shared helpers in the tests' `mod.rs`; user
    decision 2026-11-12: make every new file and test highly modular). Every function and branch gets a unit test, with stubs or fakes for
    anything external (GPU, `nvidia-smi`, the filesystem). Measure line coverage with
    `cargo llvm-cov` when it is available, record the numbers in `docs/coverage.md`, never
    claim a figure that was not measured, and raise the floor over time toward 100%. Code that
    can only run on real hardware is isolated in small files so the rest stays fully testable.
12. **Hand-off file at 98% context (user decision 2026-11-10).** Once the context window reaches
    98% (as reported by the harness or status line; if no figure is visible, when the session is
    clearly near its limit), stop starting new work and write `handoff.md` in the repository
    root, then commit and push it with any finished work. It must let a fresh agent resume cold:
    the goal; what is done and where (branch, last commit, what is pushed or merged); what was
    in progress and its exact state (uncommitted files, a running command and its output file);
    the next steps in order; open questions and decisions already taken; the commands to run
    first (build, test, coverage); and gotchas learned (tooling quirks, measurements and their
    caveats). A fresh session reads `handoff.md` first and marks it consumed (a line at the top
    saying when and by whom) before continuing. Keep it current at each phase boundary so a
    sudden stop loses little. Do not delete a handoff that has not been consumed.

    *The trigger is the status line.* A model cannot see its own context usage, so the
    project's `.claude/settings.json` runs `scripts/statusline.sh` as the status line. It reads
    the real figure from the status-line JSON (`context_window.used_percentage`, or 100 minus
    `remaining_percentage`), records it in `.superpowers/context.json`, and delegates to the
    user's own status-line script so their display is unchanged. `scripts/handoff_check.sh` runs
    as a `PostToolUse` and `UserPromptSubmit` hook; the first time the recorded figure reaches
    98% (`HANDOFF_AT` overrides) it injects an instruction to write `handoff.md` now, then
    stays quiet until usage falls under 90%. Both scripts have tests
    (`tests/tests/test_context_scripts.rs`). If that hook message appears, obey it before doing
    anything else.
10. **Stay autonomous (user decision 2026-11-10).** Whenever there is a question about how to
    proceed, continue with the **Recommended** option without asking. Make routine decisions
    by the model's own judgement and say in one line what was decided. Do not exit autonomous
    mode to ask for confirmation. Ask the user only when there is no recommended option and no
    sensible default, or for a hard blocker (needs credentials, sudo, or an action outside
    this repository). The existing limits still apply: never invent numbers, never touch
    ferrum-cpu, never force-push, and never push or merge to `main` unless the user asks.
15. **Automatic merge, cleanup and push (user decision 2026-11-11).** When a branch's work is
    committed and the full suite is green (fmt, clippy `-D warnings`, tests in all four
    configurations, coverage floor, `cargo deny`), merge it into `main` without asking (fast-forward
    or a plain merge commit, never a rebase of published history), delete the branches that are then
    fully merged or stale (local, remote and worktree), and push `main`. Push only after every
    check has passed on the merged result. Still forbidden: force-push, deleting a branch with
    unmerged commits, and merging or pushing with any failing check. This overrides the "never
    push or merge to `main`" limit in rule 10.
13. **Model size limit (user decision 2026-11-11, strict).** Never download a model larger than
    1.5B parameters, from any source or tool (`ollama pull`, `lms get`, `curl`, Hugging Face,
    anything). Models above 1.5B parameters found on this machine are deleted when found. Check
    the parameter count (model card, file name or GGUF header) before downloading; if it cannot
    be established, do not download.
14. **One coordinator and at most one helper agent (user decision 2026-11-11, strict).** The main
    session is the coordinator. At most one other agent (a subagent, fork or workflow worker) may
    run at any time: never two helpers in parallel, and no helper is started while one is still
    running. The coordinator says in one line, before starting a helper, what it will do, and keeps
    the user informed while it runs: what the helper is doing right now, in short status lines at
    every step, and what it finished or found when it returns. Nothing runs silently; if the
    helper has produced no visible progress for a while, the coordinator says what it is waiting
    for.
16. **Never pass a bare negative number to `kill` (2026-11-12).** A test that ends a process
    group calls `terminate_group` in `ferrum_gpu_gui/tests/common/mod.rs`, which runs
    `kill -s TERM -- -<pgid>` and refuses groups 0 and 1. `kill -TERM -<pgid>` without the `--` is
    forbidden: procps-ng 4.0.4 (Zorin OS 18 / Ubuntu 24.04) reads it as an option cluster and
    signals minus its first digit, so a Chrome pid starting with 1 became `kill(-1, SIGTERM)` and
    logged the whole desktop session out (nine times on 2026-11-11 and 2026-11-12; it looked like
    a GPU crash because long GPU and coverage runs burn through many pids). Signal a single
    process with `Child::kill` or a positive pid; never `pkill`/`killall` by name.
    Browser and `node` children are started through `with_deadline` (coreutils `timeout`, a hard
    lifetime that outlives a killed test run) and ended with `reap_group` (SIGTERM, then SIGKILL
    for leaders and helpers that ignore it); do not hand-roll process cleanup in tests.
17. **Test time limits (user decision 2026-11-14, replacing the 3-minute rule of the same day).**
    There is no time limit on ordinary tests. The interoperability tests (`ferrum_gpu_cli/tests/interop`,
    which drive real external tools) may take up to 10 minutes each. `scripts/test-time-limit.sh
    [GENERAL_LIMIT] [cargo args]` runs every test in its own process under GNU `timeout` and enforces
    exactly that: the general limit defaults to 0 (none; pass a number to cap ordinary tests again),
    the interop limit is 600 s (`INTEROP_LIMIT` changes it). It reports a hung test as TIMEOUT, a slow
    one with its time, and the slowest eight. libtest's own `--ensure-time` does not fire for workspace
    members, so do not rely on it. The gate runs the script and CI runs it on Linux. Tests that need an
    outside tool skip with a printed reason when it is absent. The tests that change anything outside
    the repository (a real `lms import` under `~/.lmstudio`) or use the GPU through an outside tool are
    opt-in (`FERRUM_INTEROP_LMS=1`, `FERRUM_INTEROP_GPU=1`) and log every step; see `docs/interop.md`.
18. **`metrics.md` is always current (user decision 2026-11-18).** The repository root holds `metrics.md`: size, tests,
    structure, dependencies, history, the last gate's timing and the last coverage run. It is generated, never edited by
    hand: the pre-commit hook (`scripts/githooks`, enabled with `git config core.hooksPath scripts/githooks`) refreshes it
    and adds it to every commit, `scripts/coverage.sh` rewrites its coverage section and `scripts/gate.sh` its gate
    section. In a clone or worktree where the hook is not active, run `python3 scripts/metrics.py --next-commit` before
    committing. Nothing in it is estimated; a number that cannot be read shows as "-".
