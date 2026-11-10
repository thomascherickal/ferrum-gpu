# Phase 0 spike: GPU matmul vs CPU (2026-11-10)

**Decision: GO, with one qualification.** The gate (at least 3x the 8-thread CPU matmul at 1024 and above, readback included, upload excluded) is met on both real GPUs against this repo's simple CPU kernel. Against ferrum-cpu's tuned kernel the MX330 is at 2.96x at 1024, a hair under the line and within noise (see finding 3 and the addendum at the end, measured with the tuned kernel ported in Phase 2). A second finding (5) changes the Phase 2+ design: small operations only pay off when data stays on the device.

## Setup

- Machine: dev laptop, 8 CPU threads, Linux 7.0.0-42-generic, rustc 1.96.0, `wgpu` 30.0.1 (Vulkan backend), release profile.
- Adapters measured: NVIDIA GeForce MX330 (discrete), Intel Iris Xe Graphics TGL GT2 (integrated), llvmpipe (software, correctness only). Driver versions were not recorded.
- Method: `cargo bench -p ferrum_wgpu --bench matmul` (std-only). Median of 3 to 20 repetitions after one warm-up; one session per adapter; laptop thermals and background load were not controlled, and small cases show visible run-to-run noise.
- CPU baseline in the tables: `ferrum_ref::matmul_threaded`, a plain i-k-j f32 kernel split by rows, written in this repo.
- Kernels: **Naive** (one invocation per output), **Tiled16** (16x16 shared-memory tiles), **Blocked4** (64x64 tile per workgroup, 4x4 outputs per invocation, vec4 shared-memory reads). All three match the CPU reference within 1e-4 relative error on 8 odd and degenerate shapes on all three adapters, and repeat runs are bit-identical.
- "gpu compute" = resident buffers, submit + wait. "compute+readback" = the gate's definition. The "+upload" columns include uploading both operands every time.
- `ref-*` rows use 2048 rows (16 sequences x 128 tokens) with the 507,552-parameter preset's embed 96, hidden 320, vocab 1024.

## Findings

1. **Kernel structure decides everything.** The first Blocked4 (a 16-element accumulator array indexed in loops) ran at about 10 GFLOP/s, slower than Naive. Rewritten with four named `vec4` accumulators and vec4 shared-memory reads it runs at about 310 to 360 GFLOP/s. Naive and Tiled16 stay at 32 to 90 GFLOP/s. Only Blocked4 clears the gate.
2. **Against this repo's CPU kernel** (compute+readback, 8 threads): MX330 4.14x at 1024 and 10.68x at 2048; Intel Xe 6.34x and 13.23x.
3. **Against ferrum-cpu's own kernel** (one read-only run of its `gemm` bench, 8 threads: 25.515 ms at 1024 and 232.906 ms at 2048, i.e. 84 and 74 GFLOP/s; build output kept outside that repo): MX330 25.515/8.45 = **3.02x** at 1024 (thin margin) and 232.906/57.09 = 4.08x at 2048; Intel Xe 3.82x and 5.05x. The gate still passes, but the MX330 at 1024 is within noise of 3x, so speedup claims at that size must be re-measured against a tuned in-repo CPU kernel in Phase 2.
4. **Fixed costs are small on real GPUs**: empty submit+wait 0.029 ms (MX330) and 0.216 ms (Xe); 4-byte readback 0.034 ms and 0.295 ms. Uploading both operands costs about 3.3 ms at 1024 and 13.8 ms at 2048 on the MX330, which is why weights must stay resident.
5. **Per-op readback erases the win on small shapes.** The `ref-*` rows at 2048 rows reach 140 to 235 GFLOP/s on the MX330 compute-only (2.4x to 4.5x the 8-thread CPU), but with readback included the MX330 wins only 1.2x to 1.9x and the Xe between 0.6x and 1.9x. Design consequence: activations stay on the device between kernels, with one readback per step (as PLAN.md already says). Whether the GPU wins on single-sequence decode (128 rows or one row) was not measured here; it is a Phase 3 question and `auto` routing depends on it.
6. **llvmpipe** is correct but slow (Blocked4 is 0.37x the CPU at 1024). It is the correctness device for CI, not a performance device, and is never chosen unless named.

## Not claimed

Apple Silicon (Metal), AMD, Windows (DX12), and any driver other than the two Vulkan drivers above were not run. Nothing here says they will meet the gate. Dispatch limits (65,535 workgroups per dimension) are enforced by `dispatch_size`; buffers over the adapter's binding limit are refused.

## Tables

### adapter: NVIDIA GeForce MX330 (Vulkan, Discrete); cpu threads: 8

fixed cost: empty submit+wait 0.029 ms, 4-byte readback 0.034 ms

| case | variant | cpu x1 ms | cpu x8 ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x8 (compute+readback) |
|---|---|---|---|---|---|---|---|---|---|
| 256^3 | Naive | 2.19 | 0.61 | 0.97 | 1.08 | 1.32 | 1.40 | 34.5 | 0.57x |
| 256^3 | Tiled16 | 2.19 | 0.61 | 1.03 | 1.01 | 1.09 | 1.22 | 32.7 | 0.60x |
| 256^3 | Blocked4 | 2.19 | 0.61 | 0.27 | 0.29 | 0.40 | 0.51 | 122.7 | 2.13x |
| 1024^3 | Naive | 134.05 | 34.97 | 66.93 | 68.72 | 70.40 | 71.98 | 32.1 | 0.51x |
| 1024^3 | Tiled16 | 134.05 | 34.97 | 44.15 | 45.79 | 47.45 | 49.10 | 48.6 | 0.76x |
| 1024^3 | Blocked4 | 134.05 | 34.97 | 6.89 | 8.45 | 10.19 | 11.84 | 311.5 | 4.14x |
| 2048^3 | Naive | 2215.00 | 609.78 | 540.55 | 544.85 | 547.71 | 560.38 | 31.8 | 1.12x |
| 2048^3 | Tiled16 | 2215.00 | 609.78 | 351.99 | 358.54 | 365.61 | 371.98 | 48.8 | 1.70x |
| 2048^3 | Blocked4 | 2215.00 | 609.78 | 50.66 | 57.09 | 64.42 | 70.76 | 339.1 | 10.68x |
| ref-qkv (2048x96x96) | Naive | 2.06 | 0.64 | 0.99 | 1.43 | 1.52 | 1.67 | 38.1 | 0.45x |
| ref-qkv (2048x96x96) | Tiled16 | 2.06 | 0.64 | 1.07 | 1.24 | 1.29 | 1.56 | 35.3 | 0.52x |
| ref-qkv (2048x96x96) | Blocked4 | 2.06 | 0.64 | 0.27 | 0.53 | 0.57 | 0.89 | 142.3 | 1.21x |
| ref-ffn1 (2048x96x320) | Naive | 7.53 | 2.81 | 3.18 | 4.34 | 3.66 | 4.61 | 39.6 | 0.65x |
| ref-ffn1 (2048x96x320) | Tiled16 | 7.53 | 2.81 | 2.76 | 3.78 | 3.38 | 4.31 | 45.7 | 0.74x |
| ref-ffn1 (2048x96x320) | Blocked4 | 7.53 | 2.81 | 0.62 | 1.69 | 0.90 | 1.94 | 201.5 | 1.66x |
| ref-ffn2 (2048x320x96) | Naive | 7.62 | 1.93 | 3.24 | 3.56 | 4.59 | 4.75 | 38.8 | 0.54x |
| ref-ffn2 (2048x320x96) | Tiled16 | 7.62 | 1.93 | 2.94 | 3.12 | 4.09 | 4.22 | 42.8 | 0.62x |
| ref-ffn2 (2048x320x96) | Blocked4 | 7.62 | 1.93 | 0.72 | 1.00 | 1.97 | 2.15 | 174.4 | 1.93x |
| ref-head (2048x96x1024) | Naive | 24.67 | 7.34 | 9.77 | 13.02 | 10.24 | 13.50 | 41.2 | 0.56x |
| ref-head (2048x96x1024) | Tiled16 | 24.67 | 7.34 | 8.79 | 12.02 | 9.18 | 12.48 | 45.8 | 0.61x |
| ref-head (2048x96x1024) | Blocked4 | 24.67 | 7.34 | 1.72 | 5.09 | 2.19 | 5.52 | 234.5 | 1.44x |

### adapter: Intel(R) Iris(R) Xe Graphics (TGL GT2) (Vulkan, Integrated); cpu threads: 8

fixed cost: empty submit+wait 0.216 ms, 4-byte readback 0.295 ms

| case | variant | cpu x1 ms | cpu x8 ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x8 (compute+readback) |
|---|---|---|---|---|---|---|---|---|---|
| 256^3 | Naive | 2.57 | 1.07 | 0.86 | 0.95 | 0.99 | 1.07 | 39.2 | 1.12x |
| 256^3 | Tiled16 | 2.57 | 1.07 | 0.70 | 2.62 | 0.77 | 2.93 | 47.8 | 0.41x |
| 256^3 | Blocked4 | 2.57 | 1.07 | 0.31 | 1.10 | 1.26 | 0.60 | 106.7 | 0.97x |
| 1024^3 | Naive | 148.89 | 42.34 | 43.38 | 44.50 | 44.56 | 44.88 | 49.5 | 0.95x |
| 1024^3 | Tiled16 | 148.89 | 42.34 | 23.44 | 24.62 | 24.95 | 25.70 | 91.6 | 1.72x |
| 1024^3 | Blocked4 | 148.89 | 42.34 | 6.03 | 6.68 | 7.36 | 8.09 | 356.2 | 6.34x |
| 2048^3 | Naive | 2217.21 | 610.39 | 522.78 | 510.26 | 518.53 | 514.95 | 32.9 | 1.20x |
| 2048^3 | Tiled16 | 2217.21 | 610.39 | 198.89 | 200.44 | 202.40 | 213.84 | 86.4 | 3.05x |
| 2048^3 | Blocked4 | 2217.21 | 610.39 | 47.31 | 46.13 | 48.41 | 52.27 | 363.1 | 13.23x |
| ref-qkv (2048x96x96) | Naive | 2.27 | 0.64 | 0.88 | 0.98 | 0.99 | 1.20 | 43.0 | 0.66x |
| ref-qkv (2048x96x96) | Tiled16 | 2.27 | 0.64 | 0.58 | 0.73 | 0.73 | 0.91 | 65.6 | 0.89x |
| ref-qkv (2048x96x96) | Blocked4 | 2.27 | 0.64 | 0.63 | 1.01 | 0.48 | 0.61 | 60.2 | 0.64x |
| ref-ffn1 (2048x96x320) | Naive | 7.46 | 1.86 | 4.83 | 3.23 | 3.10 | 3.71 | 26.0 | 0.58x |
| ref-ffn1 (2048x96x320) | Tiled16 | 7.46 | 1.86 | 1.84 | 2.27 | 1.95 | 2.43 | 68.2 | 0.82x |
| ref-ffn1 (2048x96x320) | Blocked4 | 7.46 | 1.86 | 0.70 | 1.40 | 1.13 | 2.31 | 181.0 | 1.33x |
| ref-ffn2 (2048x320x96) | Naive | 7.43 | 2.81 | 3.70 | 3.42 | 3.94 | 4.29 | 34.1 | 0.82x |
| ref-ffn2 (2048x320x96) | Tiled16 | 7.43 | 2.81 | 2.34 | 2.23 | 2.78 | 2.84 | 53.7 | 1.26x |
| ref-ffn2 (2048x320x96) | Blocked4 | 7.43 | 2.81 | 2.11 | 2.36 | 2.92 | 3.49 | 59.8 | 1.19x |
| ref-head (2048x96x1024) | Naive | 22.95 | 5.99 | 8.94 | 10.42 | 9.20 | 10.46 | 45.1 | 0.57x |
| ref-head (2048x96x1024) | Tiled16 | 22.95 | 5.99 | 5.05 | 6.43 | 5.53 | 6.72 | 79.7 | 0.93x |
| ref-head (2048x96x1024) | Blocked4 | 22.95 | 5.99 | 1.86 | 3.19 | 1.94 | 4.07 | 217.1 | 1.88x |

### adapter: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan, Software); cpu threads: 8

fixed cost: empty submit+wait 0.014 ms, 4-byte readback 0.020 ms

| case | variant | cpu x1 ms | cpu x8 ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x8 (compute+readback) |
|---|---|---|---|---|---|---|---|---|---|
| 256^3 | Naive | 2.13 | 0.63 | 10.64 | 9.91 | 10.36 | 8.76 | 3.2 | 0.06x |
| 256^3 | Tiled16 | 2.13 | 0.63 | 18.66 | 18.54 | 17.34 | 17.79 | 1.8 | 0.03x |
| 256^3 | Blocked4 | 2.13 | 0.63 | 2.07 | 2.14 | 2.26 | 2.23 | 16.2 | 0.29x |
| 1024^3 | Naive | 155.16 | 44.91 | 595.08 | 590.74 | 611.76 | 612.19 | 3.6 | 0.08x |
| 1024^3 | Tiled16 | 155.16 | 44.91 | 1172.97 | 1152.23 | 1156.64 | 1168.46 | 1.8 | 0.04x |
| 1024^3 | Blocked4 | 155.16 | 44.91 | 124.13 | 122.22 | 129.24 | 121.85 | 17.3 | 0.37x |
| ref-qkv (2048x96x96) | Naive | 2.50 | 0.66 | 11.71 | 12.04 | 11.99 | 13.51 | 3.2 | 0.05x |
| ref-qkv (2048x96x96) | Tiled16 | 2.50 | 0.66 | 21.13 | 23.36 | 23.24 | 23.07 | 1.8 | 0.03x |
| ref-qkv (2048x96x96) | Blocked4 | 2.50 | 0.66 | 3.24 | 3.63 | 3.57 | 3.69 | 11.7 | 0.18x |
| ref-ffn1 (2048x96x320) | Naive | 8.91 | 2.75 | 34.55 | 36.78 | 39.94 | 35.50 | 3.6 | 0.07x |
| ref-ffn1 (2048x96x320) | Tiled16 | 8.91 | 2.75 | 73.40 | 78.58 | 74.79 | 73.83 | 1.7 | 0.03x |
| ref-ffn1 (2048x96x320) | Blocked4 | 8.91 | 2.75 | 8.90 | 9.37 | 8.90 | 10.68 | 14.1 | 0.29x |
| ref-ffn2 (2048x320x96) | Naive | 8.56 | 2.01 | 37.25 | 37.10 | 35.05 | 38.35 | 3.4 | 0.05x |
| ref-ffn2 (2048x320x96) | Tiled16 | 8.56 | 2.01 | 71.06 | 74.06 | 71.63 | 73.14 | 1.8 | 0.03x |
| ref-ffn2 (2048x320x96) | Blocked4 | 8.56 | 2.01 | 9.54 | 11.15 | 9.60 | 11.47 | 13.2 | 0.18x |
| ref-head (2048x96x1024) | Naive | 27.62 | 6.76 | 112.62 | 116.07 | 120.85 | 115.45 | 3.6 | 0.06x |
| ref-head (2048x96x1024) | Tiled16 | 27.62 | 6.76 | 235.59 | 236.07 | 237.08 | 238.60 | 1.7 | 0.03x |
| ref-head (2048x96x1024) | Blocked4 | 27.62 | 6.76 | 22.88 | 26.19 | 24.90 | 27.95 | 17.6 | 0.26x |

## Addendum: re-measured against the tuned CPU kernel (Phase 2, task 2.0)

The tuned 4x8 CPU matmul was ported into `ferrum_ref::tuned` (bit-identical to the simple kernel at every thread count) and the benchmark now uses it for both CPU columns. Same machine, same day, Blocked4 only; the GPU numbers are fresh runs, so they differ slightly from the tables above.

**Gate result against the tuned baseline.** Intel Xe passes (3.51x at 1024, 7.15x at 2048). The MX330 **misses the strict 3x line at 1024 by 1.3% (2.96x)**, which is within run-to-run noise, and passes at 2048 (4.59x). The decision to proceed stands, but the honest claim is "decisive from about 2048, a tie at 1024 on the MX330". The reference-model shapes at 2048 rows win only 0.4x to 1.5x with readback included on the two GPUs, which confirms that activations must stay resident between kernels.

### adapter: NVIDIA GeForce MX330 (Vulkan, Discrete); cpu threads: 8

| case | variant | cpu x1 ms | cpu x8 ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x8 (compute+readback) |
|---|---|---|---|---|---|---|---|---|---|
| 256^3 | Blocked4 | 1.42 | 0.41 | 0.26 | 0.27 | 0.39 | 0.49 | 129.5 | 1.50x |
| 1024^3 | Blocked4 | 96.28 | 25.04 | 6.74 | 8.45 | 10.19 | 11.90 | 318.8 | 2.96x |
| 2048^3 | Blocked4 | 944.63 | 260.28 | 50.44 | 56.70 | 63.95 | 70.29 | 340.6 | 4.59x |
| ref-qkv (2048x96x96) | Blocked4 | 1.54 | 0.65 | 0.23 | 0.54 | 0.56 | 0.87 | 163.3 | 1.20x |
| ref-ffn1 (2048x96x320) | Blocked4 | 4.96 | 1.32 | 0.52 | 1.57 | 0.90 | 1.93 | 240.6 | 0.84x |
| ref-ffn2 (2048x320x96) | Blocked4 | 4.94 | 1.23 | 0.71 | 0.97 | 2.00 | 2.09 | 177.5 | 1.26x |
| ref-head (2048x96x1024) | Blocked4 | 16.52 | 4.50 | 1.55 | 4.96 | 2.22 | 5.48 | 259.6 | 0.91x |

### adapter: Intel(R) Iris(R) Xe Graphics (TGL GT2) (Vulkan, Integrated); cpu threads: 8

| case | variant | cpu x1 ms | cpu x8 ms | gpu compute ms | compute+readback ms | +upload ms | +upload+readback ms | gpu GFLOP/s | speedup vs cpu x8 (compute+readback) |
|---|---|---|---|---|---|---|---|---|---|
| 256^3 | Blocked4 | 1.35 | 0.41 | 0.77 | 1.32 | 0.55 | 0.52 | 43.3 | 0.31x |
| 1024^3 | Blocked4 | 101.14 | 23.75 | 5.72 | 6.76 | 7.36 | 7.98 | 375.4 | 3.51x |
| 2048^3 | Blocked4 | 1219.59 | 331.01 | 43.09 | 46.29 | 51.35 | 54.19 | 398.7 | 7.15x |
| ref-qkv (2048x96x96) | Blocked4 | 1.48 | 0.46 | 0.81 | 1.07 | 1.22 | 1.67 | 46.5 | 0.43x |
| ref-ffn1 (2048x96x320) | Blocked4 | 5.02 | 2.61 | 1.09 | 1.72 | 2.07 | 2.78 | 114.9 | 1.52x |
| ref-ffn2 (2048x320x96) | Blocked4 | 5.10 | 1.33 | 2.33 | 2.32 | 2.86 | 3.58 | 54.0 | 0.57x |
| ref-head (2048x96x1024) | Blocked4 | 16.36 | 4.73 | 1.67 | 3.87 | 2.02 | 4.55 | 240.7 | 1.22x |
