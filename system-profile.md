# System profile

CPU and GPU usage during the gate (`scripts/gate.sh` samples every 2 s with `scripts/system_profile.py`); do not edit by hand. A GPU column is `-` when `nvidia-smi` reported nothing. Look for rising memory or CPU peaks across rows: that is a resource regression.

```mermaid
xychart-beta
    title "CPU peak per gate run"
    x-axis ["f2ff5e1", "f2ff5e1", "b9c61d4", "b9c61d4", "b2cd83a", "60d2506"]
    y-axis "percent" 0 --> 117
    bar [83.9, 96.3, 91.9, 96.5, 91.9, 92.4]
```

<!-- profile:begin -->
| when | commit | label | seconds | CPU avg % | CPU peak % | memory peak MiB | GPU avg % | GPU peak % | GPU memory peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-25 17:28 | `60d2506` | gate | 1331 | 34.0 | 92.4 | 12310 | 1.3 | 33.0 | 207 |
| 2026-11-25 16:39 | `b2cd83a` | gate | 1517 | 39.4 | 91.9 | 11837 | 1.3 | 35.0 | 207 |
| 2026-11-23 12:10 | `b9c61d4` | gate | 1380 | 37.6 | 96.5 | 10883 | 1.3 | 34.0 | 205 |
| 2026-11-23 11:46 | `b9c61d4` | gate | 1447 | 38.3 | 91.9 | 9792 | 1.3 | 35.0 | 207 |
| 2026-11-23 10:36 | `f2ff5e1` | gate | 1563 | 40.7 | 96.3 | 10994 | 1.2 | 37.0 | 329 |
| 2026-11-23 10:09 | `f2ff5e1` | gate | 692 | 23.8 | 83.9 | 9252 | 1.5 | 37.0 | 207 |
<!-- profile:end -->

<!-- load:begin -->
Load tests, measured 2026-11-25 16:49 on `60d2506` (debug, heavy build; `scripts/load_profile.py run`). Latency is per request, from connect to the last byte; a refusal is the documented 503 past the in-flight cap.

| name | clients | requests | ok | refused | seconds | rps | p50_ms | p95_ms | p99_ms | max_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| json-server-heavy | 96 | 3840 | 3823 | 17 | 1.38 | 2775 | 34.0 | 36.2 | 48.5 | 56.6 |
| json-server | 48 | 192 | 192 | 0 | 0.14 | 1402 | 32.7 | 37.9 | 38.0 | 38.2 |
| gui-api-heavy | 96 | 3840 | 3840 | 0 | 1.22 | 3156 | 28.0 | 52.0 | 66.2 | 103.8 |
| gui-api | 32 | 192 | 192 | 0 | 0.09 | 2174 | 8.7 | 35.0 | 38.1 | 38.7 |
<!-- load:end -->
