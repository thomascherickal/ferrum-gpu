"""The hardware classes of hardware-caps.md: name, memory GB, bandwidth GB/s, note (assumptions are labelled)."""

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
