# Backlog

Hand-kept follow-ups. Every `- [ ]` line is copied into `next-steps.md` (generated); tick or delete a line here when it is done.

- [ ] Hardware rows that need a real machine: Apple Silicon, AMD, Windows (`docs/devices.md`).
- [ ] Delete the superseded unmerged branches `perf` and `quant-simplify` (`git branch -D`, needs the user's yes: banned command list).
- [ ] Security: bound the whole request time, not each read (a client sending one byte every 9 s holds a thread), in ferrum_gpu_core http read_request; test with a dribbling client.
- [ ] Security: cap the number of connection threads in the accept loop (refuse or queue past the cap), with a load test that opens more connections than the cap.
