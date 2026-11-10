# Phase 1: device runtime

Starts only after the Phase 0 gate passes.

- [ ] 1.1 `GpuTensor` (one buffer + shape), upload, readback through one helper (single place that would become async for WebGPU).
- [ ] 1.2 Pipeline/bind-group cache keyed by shader and entry point; dispatch-count helper that splits across dimensions at the granted workgroup limit.
- [ ] 1.3 Allocation inside out-of-memory error scopes mapped to errors; device-loss detection at submit.
- [ ] 1.4 `BufferPlan` wired to real allocation (plan, sum, check, allocate).

**Tests:** odd-length round trips, dispatch counts at/over the limit, plan refusal before any buffer exists; GPU-needing tests skip without an adapter.
**Gate:** suite green with and without a GPU.
