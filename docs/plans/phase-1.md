# Phase 1: device runtime

Starts only after the Phase 0 gate passes.

- [x] 1.1 `GpuTensor` (one buffer + shape), upload, readback through one helper (single place that would become async for WebGPU).
- [x] 1.2 Pipeline/bind-group cache keyed by shader and entry point; dispatch-count helper that splits across dimensions at the granted workgroup limit.
- [x] 1.3 Allocation inside out-of-memory error scopes mapped to errors; device-loss detection at submit.
- [x] 1.4 `BufferPlan` wired to real allocation (plan, sum, check, allocate).

**Tests:** odd-length round trips, dispatch counts at/over the limit, plan refusal before any buffer exists; GPU-needing tests skip without an adapter.
**Gate:** suite green with and without a GPU.

## Result (2026-11-10)

```mermaid
flowchart LR
    a["Result 2026-11-10"]:::root
    a --> n0["overview"]:::c0
    a --> n1["details"]:::c1
    a --> n2["usage"]:::c2
    a --> n3["notes"]:::c3
    classDef root fill:#dbeafe,stroke:#2563eb,color:#111
    classDef c0 fill:#dcfce7,stroke:#16a34a,color:#111
    classDef c1 fill:#fef3c7,stroke:#d97706,color:#111
    classDef c2 fill:#fce7f3,stroke:#db2777,color:#111
    classDef c3 fill:#ede9fe,stroke:#7c3aed,color:#111
```



Done. `runtime.rs` holds error scopes (validation + out-of-memory), device-loss and uncaptured-error recording (`Gpu::check`), the pipeline cache, one generic `dispatch`, `groups_for`, and `alloc_plan`. `GpuBuf` carries a shape. Gate met: 38 + 4 + 6 tests pass with no adapter (all GPU tests skip), on the MX330, on Intel Xe and on llvmpipe. A destroyed device surfaces as an error ("GPU device lost"), not a panic. Real out-of-memory could not be provoked on this box (a binding-limit-sized allocation succeeded or was refused cleanly); the OOM mapping is unit-tested instead.
