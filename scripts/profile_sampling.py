"""Part of system_profile.py."""
import datetime
from profile_host import *  # noqa: F401,F403
import os
import subprocess
import sys
import time


KEEP = 12


BEGIN, END = "<!-- profile:begin -->", "<!-- profile:end -->"


HEADER = "| when | commit | label | seconds | CPU avg % | CPU peak % | memory peak MiB | GPU avg % | GPU peak % | GPU memory peak MiB |"


def sample(path, interval=2.0):
    t0, prev = time.time(), stat_line()
    with open(path, "w") as f:
        while not os.path.exists(path + ".stop"):
            time.sleep(interval)
            cur = stat_line()
            cpu, mem, g = cpu_now(prev, cur), mem_now(), read_gpu()
            f.write(f"{time.time() - t0:.0f},{blank(cpu, 1)},{blank(mem, 0)},{g[0] if g else ''},{g[1] if g else ''}\n")
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
    return dict(seconds=float(rows[-1][0]), cpu_avg=avg(cpu), cpu_peak=max(cpu) if cpu else None, mem_peak=max(mem) if mem else None,
                gpu_avg=avg(gpu), gpu_peak=max(gpu) if gpu else None, gpu_mem_peak=max(gmem) if gmem else None)


def row(s, commit, label, when):
    f = lambda v, d=0: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    return (f"| {when} | `{commit}` | {label} | {s['seconds']:.0f} | {f(s['cpu_avg'], 1)} | {f(s['cpu_peak'], 1)} | "
            f"{f(s['mem_peak'])} | {f(s['gpu_avg'], 1)} | {f(s['gpu_peak'], 1)} | {f(s['gpu_mem_peak'])} |")
