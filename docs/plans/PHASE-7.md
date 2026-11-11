# Phase 7: verification matrix and default flip

`docs/DEVICES.md` rows (software/lavapipe, NVIDIA, Intel iGPU, Apple Silicon, AMD, Windows DX12), each with adapter, driver, OS, wgpu version, date, result, or "not run". CI: lint, deny, guard tests, suite; macOS/Windows as non-required portable jobs. Flip the default to `auto` only for classes with a green row.

## Progress (2026-11-11)

- [x] 7.1 `scripts/device-matrix.sh` runs the `ferrum_wgpu` suite once per adapter and prints a row.
- [x] 7.2 `docs/DEVICES.md`: MX330 (Vulkan), Iris Xe (Vulkan) and llvmpipe green; Mesa Iris Xe (Gl) red on the GELU parity test (4.04e-3 vs 1e-5); Apple, AMD, Windows not run.
- [x] 7.3 Default adjusted for the red row: `select::choose` never picks a `Gl` adapter unprompted (test `never_picks_an_opengl_adapter_unprompted`).
- [x] 7.4 `.github/workflows/ci.yml` (lint, deny, guard tests, suite, lavapipe row; macOS and Windows as non-required jobs). Written, not yet run on GitHub. `cargo deny check` re-run locally 2026-11-11: advisories, bans, licenses, sources ok.
- [x] 7.5 Apple Silicon, AMD and Windows DX12 rows: recorded as "not run" (no such hardware here; hosted CI runners have no GPU). Nothing is claimed for them; they stay out of the verified set for `auto`.
- [x] 7.6 Driver versions recorded in docs/DEVICES.md (NVIDIA 580.178.04, Mesa 25.2.8).

**Gate:** `auto` is flipped only for classes with a green row. Met: hardware Vulkan (NVIDIA, Intel) is green and used; OpenGL (red) and software are never auto-picked; unverified classes are not claimed. Open for later: first CI run on GitHub, and Apple, AMD and Windows hardware rows.
