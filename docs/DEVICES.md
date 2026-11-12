# Device verification matrix (Phase 7, started 2026-11-11)

One row per adapter class. A row is **green** only if the `ferrum_wgpu` test suite (180 tests: kernel parity against the CPU oracle, training, Llama/Qwen) ran on that adapter with `FERRUM_GPU_ADAPTER` set and everything passed. Produced by `scripts/device-matrix.sh`. Everything else is "not run".

What was run: `cargo test --release -p ferrum_wgpu` per adapter. What was **not** run per adapter: the other crates' tests (they use the CPU backend or the first adapter), the benchmarks (those are in [train-speed.md](train-speed.md) and [decode-crossover.md](decode-crossover.md) for the two hardware adapters), and any long training run.

Host for every row: Linux 7.0.0-42-generic x86_64, `wgpu` 30.0.1, 2026-11-11. Drivers (read from the system, not from wgpu): NVIDIA proprietary 580.178.04 (`nvidia-smi`) for the MX330; Mesa 25.2.8 (`mesa-vulkan-drivers` and `libgl1-mesa-dri`, Ubuntu 24.04 package 25.2.8-0ubuntu0.24.04.4) for the Intel Xe Vulkan row, the llvmpipe row and the Mesa OpenGL row.

| Class | Adapter (backend) | Result | Used by `auto` |
|---|---|---|---|
| NVIDIA discrete | NVIDIA GeForce MX330 (Vulkan) | **green**: 180 passed, 0 failed | yes |
| Intel integrated | Intel(R) Iris(R) Xe Graphics (TGL GT2) (Vulkan) | **green**: 180 passed, 0 failed | yes |
| Software (lavapipe) | llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan) | **green**: 180 passed, 0 failed | no, only when named |
| OpenGL on Mesa | Mesa Intel(R) Iris(R) Xe Graphics (TGL GT2) (Gl) | **red**: 178 passed, 1 failed | no, only when named |
| Apple Silicon (Metal) | not run | not run | not verified |
| AMD (Vulkan / DX12) | not run | not run | not verified |
| Windows DX12 (any vendor) | not run | not run | not verified |

## The red row

`ops::tests::gelu_matches_cpu_including_saturation` fails on the Gl adapter: max relative error 4.04e-3 against a tolerance of 1e-5. The same adapter silicon passes on Vulkan, so the difference is in the OpenGL path (its `tanh` is the likely cause, not checked). Consequence: the unprompted adapter choice (`select::choose`) now skips every adapter whose backend is `Gl`; naming one with `--gpu-adapter` still works.

## The default

`auto` still means: training on a hardware GPU, decoding on the CPU (Phase 3/4 measurements). The only change from this phase is that an OpenGL adapter, like a software one, is never picked unprompted. Apple, AMD and Windows hardware have no row, so nothing about them is claimed; their adapters are still selectable by name, as before.

## To add a row

Run `scripts/device-matrix.sh` on the machine (it lists the adapters `ferrum-gpu info` finds), paste its lines here with the driver version, and note any failing test as above.

## CI

**First CI runs (2026-11-12) found real problems, all fixed except the macOS GPU one.**
- Linux: the server panicked on a broken pipe when its parent stopped reading stdout (`println!`), which killed it mid-test. Fixed in `ferrum_gpu_server` (status lines no longer panic) and covered by a test. Linux is green.
- Windows: the tests crate and the browser-test harness used unix-only APIs, and git checked the golden fixtures out with CRLF line endings, so every byte-for-byte fixture test failed. Fixed with `cfg(unix)` gating and `fixtures/** -text` in `.gitattributes`; two tests that need `/dev/zero` or a quote in a file name are unix-only. Windows is green (whatever adapter the runner has, which is the software one, so it is not a hardware row).
- macOS (hosted `macos-latest`, Apple Silicon, a paravirtual GPU): the runner offers a Metal adapter, but **`the_gpu_backend_matches_the_cpu_backend` failed with the GPU and CPU loss curves differing by a relative 5.25 (limit 1e-3)**, and a run of the `ferrum_wgpu` kernel tests on that adapter ran for over 20 minutes without finishing, past its step timeout (it had to be cancelled). We do not know whether this is the virtualised GPU or our kernels on Metal; a real Apple machine is needed to tell. The macOS job therefore runs the suite with no adapter (`FERRUM_GPU_ADAPTER=no-such-adapter`), which tests the build and the CPU paths and is green. The Apple Silicon row in the table stays "not run".

`.github/workflows/ci.yml` (written 2026-11-11, first run 2026-11-12): the required Linux job runs `fmt --check`, clippy with `-D warnings`, `cargo deny`, the `tests` crate (independence guard, context scripts), the workspace suite, and the `ferrum_wgpu` suite on lavapipe. macOS and Windows jobs build and test but are not required, and a hosted runner has no hardware GPU, so they do not make an Apple or Windows row. Those rows stay "not run" until the suite is run on such a machine with `scripts/device-matrix.sh`.
