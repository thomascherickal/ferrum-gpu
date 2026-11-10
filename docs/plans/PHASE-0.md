# Phase 0: workspace, adapter selection, budget parsers, matmul spike

**Crates:** `ferrum_ref` (CPU reference), `ferrum_wgpu` (engine), `tests`.

## Tasks (each: tests first, one commit)

- [x] 0.1 `ferrum_ref::matmul` (copied logic: row-major f32 `[m,k]x[k,n]`, ascending-k accumulation) plus `Mat` helpers. Provenance header. Tests: known values, shape errors, non-square.
- [x] 0.2 `ferrum_wgpu::select`: `AdapterSummary { name, backend, kind, max_buffer, max_binding, max_workgroup_invocations, integrated }` and pure `choose(&[AdapterSummary], Option<&str>) -> Result<usize, SelectError>`: named substring wins (software allowed only if named), else discrete, else integrated, never software otherwise. Tests over fake lists, including empty list and unknown name (error lists what exists).
- [x] 0.3 `ferrum_wgpu::budget`: `parse_nvidia_smi`, `parse_vram_mb`, `sysfs_gpus(root)`, `Budget::resolve(override, smi, integrated, ram)`; `BufferPlan` with `check` that refuses before allocating, naming required/available bytes and any tensor over the binding limit. Tests on fixtures and temp-dir sysfs trees. No process is spawned in tests.
- [x] 0.4 `ferrum_wgpu::device`: real adapter enumeration/opening via wgpu behind the `select` types; `block_on` without extra deps. Tests skip with a printed reason when no adapter exists.
- [x] 0.5 Matmul kernels (WGSL), candidates: (a) scalar inner loop, (b) vec4 unrolled, (c) tiled with workgroup memory. Each checked against `ferrum_ref::matmul` within tolerance at odd sizes before being timed.
- [x] 0.6 `ferrum_wgpu/benches/matmul.rs` (std only): 256/1024/2048 squares and the reference model's shapes; CPU 1 thread vs 8 threads (CPU timing uses a small threaded reference written in this repo) vs GPU compute-only / +upload / +readback; fixed costs (empty submit, 4-byte readback).
- [x] 0.7 `cargo deny` with the copied allow-list against wgpu's tree.
- [x] 0.8 `docs/spike-matmul.md` with the measured table and the written go/no-go.

## Result (2026-11-10)

All tasks done. **Gate met: GO.** Blocked4 is the kernel to build on (MX330 4.14x at 1024 and 10.68x at 2048, Intel Xe 6.34x and 13.23x, against the in-repo 8-thread CPU kernel; against ferrum-cpu's tuned kernel the MX330 margin at 1024 is only 3.02x). Per-op readback erases the win on small shapes, so activations stay resident. Details and tables: [`../spike-matmul.md`](../spike-matmul.md). Adapters on the dev box: NVIDIA MX330, Intel Xe (Vulkan), llvmpipe, Mesa Intel via GL.

## Gate

At least 3x the 8-thread CPU matmul at 1024 and above on the dev box's discrete GPU, weights resident. If it fails: stop, state whether compute or fixed overhead was the bottleneck.
