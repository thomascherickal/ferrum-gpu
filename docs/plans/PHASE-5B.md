# Phase 5b: browser WebGPU

Run the same models in the browser tab, on the visitor's own GPU, with no server round trip for inference. The WGSL kernels are shared with the native engine: they are served from the Rust crate that owns them, not copied into the page.

## Decisions (taken 2026-11-11, autonomous)

| Question | Decision | Why |
|---|---|---|
| How | A small **JavaScript runtime** (`gpu_gui/web/`): GGUF reader, WebGPU device/pipeline/dispatch layer, and a `GpuLlama` port that records the same dispatch sequence as `ferrum_wgpu::models::llama`. | Compiling `ferrum_wgpu` to wasm needs an async rewrite of its blocking `block_on`/readback (every `Gpu::read` and open path), `wasm-bindgen` and a bundler; the shaders are plain WGSL and already shared, so the orchestration is the only thing to port. The cost is that the dispatch sequence exists twice; the parity test below is what keeps the copies honest. |
| Shaders | Served at `/shaders/<group>/<name>.wgsl` from a registry in `ferrum_wgpu` (`shaders::SOURCES`), with the same `gid` prelude the native runtime prepends. | One source of truth; a test fails if a shader the browser needs is missing from the registry. |
| Scope | **Inference** only: decode (`feed`) and `forwardAll`; f32 and Q8_0/Q4_0 weights. Training stays native. (`score` is not ported: the page does not use it.) | Training needs ~20 more kernels and buffer plans in JS for little gain; inference is what a visitor wants. |
| Limits | Ask for the adapter's own buffer-size limits (`maxStorageBufferBindingSize`, `maxBufferSize`) and leave everything else at the WebGPU baseline; check each tensor against the binding limit and refuse naming it (portability rule 2). Kernels must fit the 256-invocation baseline workgroup, which rules out `gemv.wgsl` (1024). | Same rules as native. |
| Verification | Parity against the CPU oracle with vectors produced by the Rust side (a seeded tiny model as GGUF + token ids + oracle logits), run in headless Chrome. | Node has no WebGPU. Headless Chrome offers only SwiftShader (software) by default, but with `--use-gl=angle --use-angle=vulkan` it also offers the real adapters (found while executing this phase), so the tests run on every distinct adapter Chrome reports and the results name which were software. |

## Tasks

- [x] 5b.1 `ferrum_wgpu::shaders`: registry of every shader (name, group, source) plus the prelude; tests: every `include_str!` kernel is registered, names are unique, all sources parse as WGSL (compiled through naga in a unit test, no GPU needed).
- [x] 5b.2 JS runtime: `gguf.js` (v3 reader, f32/f16/Q8_0/Q4_0, same block repack as `ops::quant::pack`), `webgpu.js` (adapter limits, pipeline cache, uniform padding, `dispatch1d`, readback), tested in Node where WebGPU is not needed (GGUF parsing, repack vs Rust-produced fixtures, group arithmetic).
- [x] 5b.3 `llama.js`: `GpuLlama` port (`feed`, `forwardAll`), f32 and quantised.
- [x] 5b.4 Self-test page + fixture endpoint; `gpu_gui` integration test launches headless Chrome with WebGPU enabled, collects the result over HTTP, asserts parity (skipped with a printed reason when Chrome or WebGPU is missing).
- [x] 5b.5 GUI integration: "Run in: server / this browser" switch on the Generate tab; the model comes from the same GGUF download.
- [x] 5b.6 Measurements and write-up: in-browser decode speed vs native on the same machine, adapter reported by the browser, what was and was not run.

**Gate (met 2026-11-11; see `docs/browser-webgpu.md`):** in-browser logits match the CPU oracle at every position of a short context for f32 and Q8_0 (tolerance as native, 1e-4 absolute-relative), run in headless Chrome; shaders are the native source files.

## Not in 5b

Training or fine-tuning in the browser, wasm builds of the engine, Safari/Firefox verification (Firefox WebGPU is not enabled by default on Linux), mobile browsers, WebGL fallback.

## Outcome notes

- Verified on three adapters through Chrome's ANGLE/Vulkan path (MX330, Intel Xe, SwiftShader), worst error 6.0e-6; the browser tests print a skip reason instead of failing when Node, Chrome or WebGPU is missing.
- `gemv.wgsl` (1024-invocation workgroup) is the one kernel the browser cannot use on a baseline device; single-row projections use `matmul_blocked4`.
- Browser decode is 1.2x to 1.9x slower than native for f32 and faster than native for Q8_0/Q4_0 on the medium and large shapes; both are slower than one CPU thread for f32 (numbers in `docs/browser-webgpu.md`).
