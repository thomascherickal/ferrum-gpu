#!/usr/bin/env python3
"""Back-of-envelope tables for hardware-caps.md (run: python3 -I scripts/capability-estimates.py). Every number is arithmetic on the stated inputs;
measured inputs come from today's benches (MX330, 2026-11-14), everything else is an assumption."""

MiB = 1024 * 1024
# --- measured (capability bench, one process per row, MX330, Q4_0): (params, plan MiB, ms/token)
meas_q4 = [(163e6, 279, 40.4), (409e6, 557, 62.4), (630e6, 864, 60.2), (1100e6, 956, 94.7)]
meas_q8 = [(409e6, 729, 84.5), (630e6, 1099, 85.2), (1100e6, 1449, 137.1)]
BPP = {"f32": 4.0, "Q8_0": 34 / 32, "Q4_0": 18 / 32}  # bytes per weight (block formats include scales)

def linfit(pts):
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - b * sx) / n
    return a, b

# decode time model: ms = floor + weight_MB / eff_GBps  (weight bytes = params * bpp)
pts = [(p * BPP["Q4_0"] / 1e6, ms) for p, _, ms in meas_q4]  # (MB, ms)
floor_ms, ms_per_mb = linfit(pts)
eff_gbps = 1.0 / ms_per_mb  # MB/ms == GB/s
pts8 = [(p * BPP["Q8_0"] / 1e6, ms) for p, _, ms in meas_q8]
f8, s8 = linfit(pts8)

out = []
w = out.append
w("### Fit of the decode time on the MX330 (from the measured rows)\n")
w(f"- Q4_0 rows, time = floor + weight bytes / bandwidth: floor **{floor_ms:.0f} ms**, effective bandwidth **{eff_gbps:.1f} GB/s** (4 points).")
w(f"- Q8_0 rows (3 points): floor {f8:.0f} ms, effective bandwidth {1/s8:.1f} GB/s.")
w("- The MX330's memory is 64-bit GDDR5; its published peak is about 56 GB/s (product specification, not measured here).")
w(f"  So the engine reaches roughly {eff_gbps/56*100:.0f}% of the peak, plus a fixed cost per token (about 30 layers of small dispatches).\n")

# --- model overhead: M(P) = P*bpp*1.10 + 0.30 GB  (f32 embedding, f32 KV at ctx 2048, scratch), check vs measured plans
w("### Check of the memory formula against the measured plans\n")
w("`M = params x bytes/weight x 1.10 + 300 MB` (inference). Plans are `ferrum_wgpu::llama::plan` totals.\n")
w("| shape | params | precision | formula MB | measured plan MB | error |")
w("|---|---|---|---|---|---|")
rows = [("SmolLM-135M", 163e6, "Q4_0", 279), ("SmolLM-135M", 163e6, "Q8_0", 343), ("SmolLM-135M", 163e6, "f32", 712),
        ("SmolLM-360M", 409e6, "Q4_0", 557), ("SmolLM-360M", 409e6, "Q8_0", 729), ("SmolLM-360M", 409e6, "f32", 1721),
        ("Qwen2-0.5B", 630e6, "Q4_0", 864), ("Qwen2-0.5B", 630e6, "Q8_0", 1099),
        ("TinyLlama-1.1B", 1100e6, "Q4_0", 956), ("TinyLlama-1.1B", 1100e6, "Q8_0", 1449), ("TinyLlama-1.1B", 1100e6, "f32", 4286)]
errs = []
for n, p, q, plan in rows:
    f = (p * BPP[q] * 1.10 + 300e6) / 1e6; m = plan * MiB / 1e6; e = (f - m) / m * 100; errs.append(e)
    w(f"| {n} | {p/1e6:.0f}M | {q} | {f:.0f} | {m:.0f} | {e:+.0f}% |")
w(f"\nError range {min(errs):+.0f}% to {max(errs):+.0f}%. Under-estimates are the big-vocabulary small models (the f32 embedding is large next to the rest).\n")

def pmax(mem_gb, frac, prec, train=False):
    usable = mem_gb * 1e9 * frac
    if train:  # f32 weights + grads + two Adam moments = 16 B/param, +8% for activations at small batch (measured: 2.6% to 8%)
        return max(0.0, (usable - 300e6) / (16 * 1.08)) / 1e9
    return max(0.0, (usable - 300e6) / (BPP[prec] * 1.10)) / 1e9

def fmt(b):
    return f"{b:.1f}B" if b >= 1 else f"{b*1000:.0f}M"

w("### Largest model by memory (billions of parameters)\n")
w("Discrete cards: 90% of VRAM usable (display and driver take the rest; the headless MX330 showed 1989 of 2048 MiB free, 97%). Unified memory: 70% usable (the GPU's share of the machine; an assumption, see the text). Inference at context 2048; training is f32 AdamW, batch 1.\n")
w("| memory | usable (discrete 90% / unified 70%) | infer Q4_0 | infer Q8_0 | infer f32 | train f32 (full fine-tune) |")
w("|---|---|---|---|---|---|")
for g in [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]:
    d = lambda pr, t=False: fmt(pmax(g, .90, pr, t))
    u = lambda pr, t=False: fmt(pmax(g, .70, pr, t))
    w(f"| {g} GB | {g*.9:.1f} / {g*.7:.1f} GB | {d('Q4_0')} / {u('Q4_0')} | {d('Q8_0')} / {u('Q8_0')} | {d('f32')} / {u('f32')} | {d('f32', True)} / {u('f32', True)} |")
w("\nEach cell is `discrete / unified`.\n")

# --- speed: roofline and ferrum-gpu estimate
LAYERS = {0.5: 24, 1.5: 28, 7: 32, 13: 40, 34: 48, 70: 80}  # typical depth per size class
FIT_LAYERS = (30 + 32 + 24 + 22) / 4  # layers of the four fitted shapes
per_layer_ms = floor_ms / FIT_LAYERS

def tps(p_b, bw, prec="Q4_0"):
    mb = p_b * 1e9 * BPP[prec] / 1e6
    roof = bw * 1000 / mb  # GB/s -> MB/ms ; tokens/s = 1000/(mb/(bw)) ...
    roof = 1000.0 / (mb / bw)  # bw GB/s = MB/ms -> ms = mb/bw
    est = 1000.0 / (per_layer_ms * LAYERS[p_b] + mb / (eff_gbps * bw / 56.0))
    return roof, est

hw = [  # name, memory GB, bandwidth GB/s, note
 ("MX330 (this machine, measured)", 2, 56, "discrete, 64-bit GDDR5, published peak"),
 ("Entry laptop dGPU, 4 GB", 4, 112, "assumed class (128-bit GDDR5/6 around 112 to 128)"),
 ("Mid dGPU, 8 GB", 8, 256, "assumed class (GDDR6, 128-bit to 256-bit)"),
 ("Upper-mid dGPU, 16 GB", 16, 448, "assumed class (GDDR6/6X, 256-bit)"),
 ("High dGPU, 24 GB", 24, 936, "class of the RTX 3090 and 4090 (936 to 1008 GB/s, published)"),
 ("Top dGPU, 32 GB", 32, 1792, "class of the RTX 5090 (published, from memory)"),
 ("x86 iGPU, DDR4-3200 dual channel", 16, 51.2, "3200 MT/s x 16 B"),
 ("x86 iGPU, DDR5-5600 dual channel", 32, 89.6, "5600 MT/s x 16 B"),
 ("Apple M1, 16 GB", 16, 68, "published 68.25 GB/s"),
 ("Apple M-series base (M2/M3), 24 GB", 24, 100, "published 100 GB/s"),
 ("Apple M4, 32 GB", 32, 120, "published 120 GB/s"),
 ("Apple M-series Pro, 48 GB", 48, 273, "M4 Pro published 273 GB/s"),
 ("Apple M-series Max, 128 GB", 128, 546, "M4 Max published 546 GB/s (top bin)"),
 ("Apple M-series Ultra, 192 GB", 192, 819, "M3 Ultra published 819 GB/s"),
 ("Hypothetical M5-class Max, 128 GB", 128, 600, "ASSUMPTION: Max-class bandwidth near 600 GB/s; no M5 Max spec was checked"),
 ("Hypothetical M5-class Ultra, 256 GB", 256, 1200, "ASSUMPTION: two Max dies, 2 x 600 GB/s"),
 ("LPDDR5X-8000 256-bit unified APU, 128 GB", 128, 256, "8000 MT/s x 32 B (Strix-Halo-class)"),
 ("Server 8-channel DDR5-4800, 512 GB", 512, 307, "4800 MT/s x 64 B (one socket)"),
 ("Server 2 x 12-channel DDR5-4800, 1 TB+", 1024, 921.6, "2 x 4800 MT/s x 96 B; NUMA cuts this in practice"),
]
w("### Decode speed (tokens per second, single stream) for a model that fits\n")
w(f"Ceiling = bandwidth / weight bytes (each token reads every weight once). ferrum-gpu estimate = 1 / (layers x {per_layer_ms:.1f} ms + weight bytes / effective bandwidth), layers 24/28/32/40/48/80 for 0.5/1.5/7/13/34/70B, with effective bandwidth scaled from the MX330 fit ({eff_gbps:.1f} GB/s at 56 GB/s peak, i.e. {eff_gbps/56*100:.0f}%). Q4_0 weights. **The estimate is calibrated on one card and extrapolated; no other row was run.**\n")
sizes = [0.5, 1.5, 7, 13, 34, 70]
w("| hardware | memory | bandwidth GB/s | " + " | ".join(f"{s}B ceiling / est" for s in sizes) + " |")
w("|---|---|---|" + "---|" * len(sizes))
for n, g, bw, note in hw:
    cells = []
    for s in sizes:
        need = (s * 1e9 * BPP['Q4_0'] * 1.10 + 300e6) / 1e9
        frac = 0.70 if n.startswith(("Apple", "Hypothetical", "LPDDR", "x86", "Server")) else 0.90
        if n.startswith("MX330") and s == 1.5:
            cells.append("OOM (measured: 1.78B shape)")
        elif need > g * frac:
            cells.append("no fit")
        else:
            r, e = tps(s, bw); cells.append(f"{r:.0f} / {e:.1f}")
    w(f"| {n} | {g} GB | {bw:g} | " + " | ".join(cells) + " |")
w("\n" + "\n".join(f"- {n}: {note}" for n, _g, _bw, note in hw) + "\n")
print("\n".join(out))
