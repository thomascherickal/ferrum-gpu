"""Part of system_profile.py."""
import datetime
import os
import subprocess
import sys
import time


KEEP = 12


BEGIN, END = "<!-- profile:begin -->", "<!-- profile:end -->"


HEADER = "| when | commit | label | seconds | CPU avg % | CPU peak % | memory peak MiB | GPU avg % | GPU peak % | GPU memory peak MiB |"


def cpu_busy(a, b):
    """CPU busy % between two `/proc/stat` first lines (user nice system idle iowait ...)."""
    x = [int(v) for v in a.split()[1:]]
    y = [int(v) for v in b.split()[1:]]
    idle = (y[3] + y[4]) - (x[3] + x[4])
    total = sum(y) - sum(x)
    return 100.0 * (total - idle) / total if total > 0 else 0.0


def mem_used_mib(meminfo):
    v = {k: int(rest.split()[0]) for k, rest in (line.split(":", 1) for line in meminfo.splitlines() if ":" in line)}
    return (v["MemTotal"] - v["MemAvailable"]) / 1024.0


def gpu_reading(text):
    """(busy %, memory MiB) of the busiest GPU in `nvidia-smi --query-gpu=utilization.gpu,memory.used`."""
    best = None
    for line in text.strip().splitlines():
        try:
            u, m = (float(x) for x in line.replace("%", "").replace("MiB", "").split(","))
        except ValueError:
            continue
        best = (u, m) if best is None or u > best[0] else best
    return best


def read_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return gpu_reading(out)


def stat_line():
    with open("/proc/stat") as f:
        return f.readline()


def sample(path, interval=2.0):
    t0, prev = time.time(), stat_line()
    with open(path, "w") as f:
        while not os.path.exists(path + ".stop"):
            time.sleep(interval)
            cur = stat_line()
            with open("/proc/meminfo") as m:
                mem = mem_used_mib(m.read())
            g = read_gpu()
            f.write(f"{time.time() - t0:.0f},{cpu_busy(prev, cur):.1f},{mem:.0f},{g[0] if g else ''},{g[1] if g else ''}\n")
            f.flush()
            prev = cur


def summarize(text):
    """{seconds, cpu_avg, cpu_peak, mem_peak, gpu_avg, gpu_peak, gpu_mem_peak} from sample lines."""
    rows = [l.split(",") for l in text.strip().splitlines() if l.count(",") == 4]
    if not rows:
        return None
    col = lambda i: [float(r[i]) for r in rows if r[i] != ""]  # noqa: E731
    cpu, mem, gpu, gmem = col(1), col(2), col(3), col(4)
    avg = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return dict(seconds=float(rows[-1][0]), cpu_avg=avg(cpu), cpu_peak=max(cpu), mem_peak=max(mem),
                gpu_avg=avg(gpu), gpu_peak=max(gpu) if gpu else None, gpu_mem_peak=max(gmem) if gmem else None)


def row(s, commit, label, when):
    f = lambda v, d=0: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    return (f"| {when} | `{commit}` | {label} | {s['seconds']:.0f} | {f(s['cpu_avg'], 1)} | {f(s['cpu_peak'], 1)} | "
            f"{f(s['mem_peak'])} | {f(s['gpu_avg'], 1)} | {f(s['gpu_peak'], 1)} | {f(s['gpu_mem_peak'])} |")
