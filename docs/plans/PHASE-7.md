# Phase 7: verification matrix and default flip

`docs/DEVICES.md` rows (software/lavapipe, NVIDIA, Intel iGPU, Apple Silicon, AMD, Windows DX12), each with adapter, driver, OS, wgpu version, date, result, or "not run". CI: lint, deny, guard tests, suite; macOS/Windows as non-required portable jobs. Flip the default to `auto` only for classes with a green row.

## Progress (2026-11-11)

- [x] 7.1 `scripts/device-matrix.sh` runs the `ferrum_wgpu` suite once per adapter and prints a row.
- [x] 7.2 `docs/DEVICES.md`: MX330 (Vulkan), Iris Xe (Vulkan) and llvmpipe green; Mesa Iris Xe (Gl) red on the GELU parity test (4.04e-3 vs 1e-5); Apple, AMD, Windows not run.
- [x] 7.3 Default adjusted for the red row: `select::choose` never picks a `Gl` adapter unprompted (test `never_picks_an_opengl_adapter_unprompted`).
- [ ] 7.4 CI workflow (lint, deny, guard tests, suite; macOS and Windows as non-required jobs). `cargo deny check` has not been re-run in this phase.
- [ ] 7.5 Rows for Apple Silicon, AMD and Windows DX12: need that hardware, or a CI runner that has it.
- [ ] 7.6 Driver versions in each row (not captured by the script yet).
