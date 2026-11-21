# Backlog

Hand-kept follow-ups. Every `- [ ]` line is copied into `next-steps.md` (generated); tick or delete a line here when it is done.

- [ ] GUI self-test: choose the packing kind from the file's real tensor type (`QKind::from_ggml`) instead of from the requested precision; this closes the last uncovered GUI lines.
- [ ] Performance and load tests: a load test for the JSON server (many clients, in-flight cap, latency percentiles) and for the GUI API; record the figures in `system-profile.md`.
- [ ] Security tests: path handling of the file API and the jobs allow-list, request-size and header limits, token handling (AGENTS.md section 1).
- [ ] Delete branches and worktrees that are fully merged or superseded (`perf`, `perf-final`, `metrics`, `fast-gate`, `quant-formats`, `quant-simplify`, `gui-features`, `worktree-gui-studio`) once `main` contains their work.
- [ ] Hardware rows that need a real machine: Apple Silicon, AMD, Windows (`docs/devices.md`).
