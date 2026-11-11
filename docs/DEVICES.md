# Device verification matrix (Phase 7, started 2026-11-11)

One row per adapter class. A row is **green** only if the `ferrum_wgpu` test suite (180 tests: kernel parity against the CPU oracle, training, Llama/Qwen) ran on that adapter with `FERRUM_GPU_ADAPTER` set and everything passed. Produced by `scripts/device-matrix.sh`. Everything else is "not run".

What was run: `cargo test --release -p ferrum_wgpu` per adapter. What was **not** run per adapter: the other crates' tests (they use the CPU backend or the first adapter), the benchmarks (those are in [train-speed.md](train-speed.md) and [decode-crossover.md](decode-crossover.md) for the two hardware adapters), and any long training run.

Host for every row: Linux 7.0.0-42-generic x86_64, `wgpu` 30.0.1, 2026-11-11.

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
