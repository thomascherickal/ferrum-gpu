# Phase 7: verification matrix and default flip

`docs/DEVICES.md` rows (software/lavapipe, NVIDIA, Intel iGPU, Apple Silicon, AMD, Windows DX12), each with adapter, driver, OS, wgpu version, date, result, or "not run". CI: lint, deny, guard tests, suite; macOS/Windows as non-required portable jobs. Flip the default to `auto` only for classes with a green row.
